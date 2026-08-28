"""Autoregressive dual-pointer CVRP decoder.

The decoder consumes fused node embeddings and constructs a full CVRP solution one node at a
time. Feasibility is enforced by the decoder itself, not by the diffusion prior: a visited mask
removes served customers, a capacity mask removes customers whose demand exceeds the vehicle's
remaining load, and the depot is selectable whenever the vehicle is away from it, which closes the
current route and reloads.

The predicted route-membership matrix ``M_hat`` only biases decoding through the local pointer, so
a wrong prior can cost tour quality but can never produce an infeasible route.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from vrp_diffusion_quantum.models.fusion_encoder import FusionEncoder, FusionEncoderOutput
from vrp_diffusion_quantum.models.global_encoder import GlobalEncoder, GlobalEncoderOutput
from vrp_diffusion_quantum.models.local_masked_encoder import (
    AdjacencyMode,
    LocalMaskedEncoder,
    LocalMaskedEncoderOutput,
    build_local_attention_prior,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NSTART_CAP",
    "CVRPPolicy",
    "DecodeMode",
    "DecoderRollout",
    "DualPointerDecoder",
    "PolicyEncoding",
    "actions_to_routes",
    "build_decoder_local_adjacency",
    "nstart_count",
    "repeat_encoding",
    "select_nstart_nodes",
    "select_start_nodes",
]

DecodeMode = Literal["greedy", "sampling"]

# CMD Algorithm 1: NStart is every customer when N <= 100, else the 100 closest to the depot.
NSTART_CAP = 100

_CAPACITY_TOLERANCE = 1e-6
_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class DecoderRollout:
    """One batched construction rollout.

    ``actions`` holds node indices in visit order, starting after the implicit depot position; a
    depot entry closes the current route. ``log_probability`` sums the log-probabilities of the
    chosen actions, which is what REINFORCE differentiates. ``cost`` is the Euclidean tour length
    including every depot return.
    """

    actions: Tensor  # [batch, steps] int64 node indices
    log_probability: Tensor  # [batch]
    cost: Tensor  # [batch]
    entropy: Tensor  # [batch] mean per-step entropy of the action distribution


@dataclass(frozen=True)
class PolicyEncoding:
    """Instance-level tensors computed once per batch and reused by every rollout."""

    node_embeddings: Tensor  # [batch, n_nodes, embedding_dim]
    graph_embedding: Tensor  # [batch, embedding_dim]
    coords: Tensor  # [batch, n_nodes, 2]
    demands: Tensor  # [batch, n_nodes]
    capacity: Tensor  # [batch]
    depot_index: Tensor  # [batch]
    node_mask: Tensor  # [batch, n_nodes] bool
    local_adjacency: Tensor | None  # [batch, n_nodes, n_nodes] bool
    # Only populated by the gated fusion encoder; see FusionEncoderOutput.fusion_gate. Kept on the
    # encoding so training can watch whether the local branch is actually contributing.
    fusion_gate: Tensor | None = None


def build_decoder_local_adjacency(
    m_hat: Tensor,
    customer_node_indices: Tensor,
    customer_mask: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    """Map customer-only ``M_hat`` to a boolean node-level neighbourhood for the local pointer.

    Reuses :func:`build_local_attention_prior` so the decoder and the local encoder read the same
    prior. The depot stays reachable from every customer, otherwise a sparse ``M_hat`` could leave
    the local pointer unable to close a route.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    prior = build_local_attention_prior(
        m_hat,
        customer_node_indices,
        customer_mask,
        depot_index,
        node_mask,
    )
    return prior.allowed_pairs & (prior.weights >= threshold)


def nstart_count(node_mask: Tensor, *, cap: int = NSTART_CAP) -> int:
    """``NStart`` count from CMD Algorithm 1: ``N`` when ``N <= cap``, otherwise ``cap``.

    Which customers those starts are is :func:`select_nstart_nodes`. A mixed-size padded batch uses
    the smallest ``N`` so every rollout is a distinct start rather than a wrap-around duplicate.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    customer_counts = node_mask.to(dtype=torch.bool).sum(dim=1) - 1
    smallest = int(customer_counts.min().item())
    if smallest < 1:
        raise ValueError("every instance must contain at least one customer")
    return min(smallest, cap)


def select_nstart_nodes(
    coords: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
    *,
    cap: int = NSTART_CAP,
) -> Tensor:
    """CMD Algorithm 1 ``NStart``.

    Every customer if ``N <= cap``, otherwise the ``cap`` customers closest to the depot.
    """
    return select_start_nodes(coords, depot_index, node_mask, nstart_count(node_mask, cap=cap))


def select_start_nodes(
    coords: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
    num_starts: int,
) -> Tensor:
    """Pick ``num_starts`` customers closest to the depot.

    :func:`select_nstart_nodes` is the paper rule. This helper is what that function (and a smaller
    ``num_starts`` override) call: closest-first so the ``N > cap`` branch keeps the nearest
    customers, and a reduced count stays a depot-biased subset rather than an arbitrary slice.

    When an instance has fewer customers than ``num_starts`` the selection wraps around; duplicate
    starts stay valid rollouts and only cost redundant compute.
    """
    if num_starts < 1:
        raise ValueError(f"num_starts must be >= 1, got {num_starts}")
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [batch, n_nodes, 2], got {tuple(coords.shape)}")
    batch_size, n_nodes, _ = coords.shape
    mask = node_mask.to(dtype=torch.bool)
    depot_long = depot_index.to(dtype=torch.long)
    rows = torch.arange(batch_size, device=coords.device)

    depot_coords = coords[rows, depot_long]
    distance = torch.linalg.norm(coords - depot_coords[:, None, :], dim=-1)
    is_customer = mask.clone()
    is_customer[rows, depot_long] = False
    if not bool(is_customer.any(dim=1).all()):
        raise ValueError("every instance must contain at least one customer")

    distance = distance.masked_fill(~is_customer, float("inf"))
    order = torch.argsort(distance, dim=1)
    customer_counts = is_customer.sum(dim=1, keepdim=True)
    offsets = torch.arange(num_starts, device=coords.device)[None, :]
    positions = (offsets % customer_counts).clamp(max=n_nodes - 1)
    return torch.gather(order, 1, positions)


def actions_to_routes(
    actions: Tensor,
    depot_index: Tensor,
    customer_node_indices: Tensor,
    customer_mask: Tensor,
) -> list[list[list[int]]]:
    """Convert node-index action sequences into per-instance customer-id route lists."""
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [batch, steps], got {tuple(actions.shape)}")
    batch_size = actions.shape[0]
    if depot_index.shape[0] != batch_size or customer_node_indices.shape[0] != batch_size:
        raise ValueError("actions, depot_index, and customer_node_indices must share a batch dim")

    action_list = actions.detach().cpu().tolist()
    depot_list = depot_index.detach().cpu().tolist()
    node_indices = customer_node_indices.detach().cpu().tolist()
    real_customers = customer_mask.detach().cpu().to(dtype=torch.bool).tolist()

    all_routes: list[list[list[int]]] = []
    for index in range(batch_size):
        node_to_customer = {
            int(node): customer_id
            for customer_id, node in enumerate(node_indices[index])
            if real_customers[index][customer_id]
        }
        depot = int(depot_list[index])
        routes: list[list[int]] = []
        current: list[int] = []
        for node in action_list[index]:
            node_id = int(node)
            if node_id == depot:
                if current:
                    routes.append(current)
                    current = []
                continue
            if node_id in node_to_customer:
                current.append(node_to_customer[node_id])
        if current:
            routes.append(current)
        all_routes.append(routes)
    return all_routes


def _uniform_sample(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """``torch.rand`` that tolerates a CPU generator paired with CUDA tensors."""
    if generator is not None and torch.device(generator.device).type != device.type:
        drawn = torch.rand(*shape, generator=generator, device=generator.device)
        return drawn.to(device=device, dtype=dtype)
    return torch.rand(*shape, generator=generator, device=device, dtype=dtype)


def _pairwise_savings_bias(
    coords: Tensor,
    depot_index: Tensor,
    *,
    epsilon: float,
    depot_bias: Tensor,
) -> Tensor:
    """Clarke-Wright savings bias ``log(dist(0,i) + dist(0,j) - dist(i,j))`` (CMD eq. 25).

    The triangle inequality keeps savings non-negative, but nearly collinear triples drive them to
    zero, so the term is clamped before the logarithm. Choosing the depot as the next node saves
    exactly nothing, which would make the logarithm diverge, so the depot column carries a learned
    scalar instead: the eagerness to close a route is a decision for training, not a constant.
    """
    batch_size, n_nodes, _ = coords.shape
    rows = torch.arange(batch_size, device=coords.device)
    depot_long = depot_index.to(dtype=torch.long)
    depot_coords = coords[rows, depot_long]
    depot_distance = torch.linalg.norm(coords - depot_coords[:, None, :], dim=-1)
    pair_distance = torch.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], dim=-1)
    savings = depot_distance[:, :, None] + depot_distance[:, None, :] - pair_distance
    bias = torch.log(savings.clamp_min(epsilon))
    depot_column = depot_long.view(batch_size, 1, 1).expand(batch_size, n_nodes, 1)
    return bias.scatter(2, depot_column, depot_bias.expand(batch_size, n_nodes, 1))


class _PointerAttention(nn.Module):
    """Multi-head glimpse from one context query over the available nodes (CMD eqs. 22-23)."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def project_keys_values(self, static_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Project static node embeddings once per instance into per-head keys and values."""
        batch_size, n_nodes, _ = static_embeddings.shape

        def split(layer: nn.Linear) -> Tensor:
            projected: Tensor = layer(static_embeddings)
            reshaped = projected.view(batch_size, n_nodes, self.num_heads, self.head_dim)
            return reshaped.transpose(1, 2)

        return split(self.key_projection), split(self.value_projection)

    def forward(
        self,
        context: Tensor,
        keys: Tensor,
        values: Tensor,
        available: Tensor,
    ) -> Tensor:
        """Attend from ``context`` ``[batch, dim]`` to the nodes flagged in ``available``."""
        batch_size, embedding_dim = context.shape
        projected: Tensor = self.query_projection(context)
        query = projected.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~available[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, values)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, embedding_dim)
        output: Tensor = self.output_projection(attended)
        return output


class DualPointerDecoder(nn.Module):
    """Construct CVRP routes autoregressively with fused local and global pointers.

    The local pointer attends only inside the ``M_hat`` neighbourhood of the current node, the
    global pointer attends over every available node, and their glimpses are summed into a single
    context vector (CMD eq. 21). A single-head layer then scores candidates, optionally biased by
    the Clarke-Wright savings term, and the scores are clipped with ``C * tanh(.)`` (eq. 26).
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        num_heads: int = 8,
        clip_constant: float = 10.0,
        use_local_pointer: bool = True,
        use_global_pointer: bool = True,
        use_savings_bias: bool = True,
        use_context_perception: bool = True,
        savings_weight: float = 1.0,
        savings_epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        if clip_constant <= 0:
            raise ValueError(f"clip_constant must be positive, got {clip_constant}")
        if not use_local_pointer and not use_global_pointer:
            raise ValueError("at least one of use_local_pointer/use_global_pointer must be enabled")
        if savings_weight < 0:
            raise ValueError(f"savings_weight must be >= 0, got {savings_weight}")
        if not 0.0 < savings_epsilon <= 1.0:
            raise ValueError(f"savings_epsilon must be in (0, 1], got {savings_epsilon}")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.clip_constant = clip_constant
        self.use_local_pointer = use_local_pointer
        self.use_global_pointer = use_global_pointer
        self.use_savings_bias = use_savings_bias
        self.use_context_perception = use_context_perception
        self.savings_weight = savings_weight
        self.savings_epsilon = savings_epsilon

        self.static_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.capacity_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.global_pointer = (
            _PointerAttention(embedding_dim, num_heads) if use_global_pointer else None
        )
        self.local_pointer = (
            _PointerAttention(embedding_dim, num_heads) if use_local_pointer else None
        )
        self.context_perception = (
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            if use_context_perception
            else None
        )
        self.score_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.score_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.depot_savings_bias = nn.Parameter(torch.zeros(())) if use_savings_bias else None

    def forward(
        self,
        encoding: PolicyEncoding,
        *,
        decode_mode: DecodeMode = "greedy",
        generator: torch.Generator | None = None,
        start_nodes: Tensor | None = None,
    ) -> DecoderRollout:
        """Roll out one full solution per batch element.

        Args:
            encoding: per-instance embeddings and geometry from :class:`CVRPPolicy`.
            decode_mode: ``greedy`` takes the arg-max action, ``sampling`` draws from the policy.
            generator: RNG used by ``sampling``; pass one for reproducible rollouts.
            start_nodes: optional ``[batch]`` node indices forced as the first visit, which is how
                CMD ``NStart`` (and a smaller multi-start override) diversifies rollouts.
        """
        if decode_mode not in ("greedy", "sampling"):
            raise ValueError(f"decode_mode must be 'greedy' or 'sampling', got {decode_mode!r}")

        state = _RolloutState.create(encoding, self.embedding_dim)
        static: Tensor = self.static_projection(encoding.node_embeddings)
        static = static * state.node_mask.unsqueeze(-1).to(dtype=static.dtype)
        score_keys: Tensor = self.score_key(static)

        pointer_cache: dict[str, tuple[Tensor, Tensor]] = {}
        if self.global_pointer is not None:
            pointer_cache["global"] = self.global_pointer.project_keys_values(static)
        if self.local_pointer is not None:
            pointer_cache["local"] = self.local_pointer.project_keys_values(static)

        savings_bias: Tensor | None = None
        if self.depot_savings_bias is not None:
            savings_bias = _pairwise_savings_bias(
                state.coords,
                state.depot_index,
                epsilon=self.savings_epsilon,
                depot_bias=self.depot_savings_bias.to(dtype=state.dtype),
            )
        if self.use_local_pointer and encoding.local_adjacency is None:
            logger.debug("local pointer active without M_hat; using global availability instead")

        depot_embedding = static[state.rows, state.depot_index]
        graph_embedding = encoding.graph_embedding.to(dtype=state.dtype)
        actions: list[Tensor] = []

        for step in range(state.max_steps):
            action_mask = state.action_mask()
            if step == 0 and start_nodes is not None:
                action = start_nodes.to(device=state.device, dtype=torch.long)
                if not bool(action_mask.gather(1, action[:, None]).all()):
                    raise ValueError("start_nodes must be feasible customers at the first step")
                step_log_probability = torch.zeros_like(state.cost)
                step_entropy = torch.zeros_like(state.cost)
            else:
                logits = self._action_logits(
                    state=state,
                    static=static,
                    score_keys=score_keys,
                    pointer_cache=pointer_cache,
                    depot_embedding=depot_embedding,
                    graph_embedding=graph_embedding,
                    action_mask=action_mask,
                    savings_bias=savings_bias,
                )
                probabilities = torch.softmax(logits, dim=-1)
                action = self._select_action(
                    probabilities,
                    decode_mode=decode_mode,
                    generator=generator,
                    action_mask=action_mask,
                )
                chosen = probabilities.gather(1, action[:, None]).squeeze(1)
                step_log_probability = torch.log(chosen.clamp_min(_PROBABILITY_FLOOR))
                floored = probabilities.clamp_min(_PROBABILITY_FLOOR)
                step_entropy = -(probabilities * torch.log(floored)).sum(dim=-1)

            actions.append(state.step(action, step_log_probability, step_entropy))
            if bool(state.finished.all()):
                break

        if not bool(state.finished.all()):
            raise RuntimeError(
                f"decoder did not finish every instance within {state.max_steps} steps; "
                "this indicates an inconsistent visited or capacity mask"
            )

        return DecoderRollout(
            actions=torch.stack(actions, dim=1),
            log_probability=state.log_probability,
            cost=state.final_cost(),
            entropy=state.mean_entropy(),
        )

    def _action_logits(
        self,
        *,
        state: _RolloutState,
        static: Tensor,
        score_keys: Tensor,
        pointer_cache: dict[str, tuple[Tensor, Tensor]],
        depot_embedding: Tensor,
        graph_embedding: Tensor,
        action_mask: Tensor,
        savings_bias: Tensor | None,
    ) -> Tensor:
        """Score every node for the current step (CMD eqs. 20-26)."""
        last_embedding = static[state.rows, state.current_node]
        capacity_ratio = state.capacity_ratio().unsqueeze(-1)
        capacity_embedding: Tensor = self.capacity_projection(capacity_ratio)
        local_context = depot_embedding + last_embedding + capacity_embedding
        global_context = local_context + graph_embedding

        context = torch.zeros_like(local_context)
        if self.global_pointer is not None:
            keys, values = pointer_cache["global"]
            context = context + self.global_pointer(global_context, keys, values, action_mask)
        if self.local_pointer is not None:
            keys, values = pointer_cache["local"]
            context = context + self.local_pointer(
                local_context, keys, values, state.local_available(action_mask)
            )
        if self.context_perception is not None:
            perception: Tensor = self.context_perception(local_context)
            context = context + perception

        query: Tensor = self.score_query(context)
        logits = torch.einsum("bd,bnd->bn", query, score_keys) / math.sqrt(self.embedding_dim)
        if savings_bias is not None:
            logits = logits + self.savings_weight * savings_bias[state.rows, state.current_node]
        logits = self.clip_constant * torch.tanh(logits)
        return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

    def _select_action(
        self,
        probabilities: Tensor,
        *,
        decode_mode: DecodeMode,
        generator: torch.Generator | None,
        action_mask: Tensor,
    ) -> Tensor:
        greedy = torch.argmax(probabilities, dim=-1)
        if decode_mode == "greedy":
            return greedy
        uniform = _uniform_sample(
            (probabilities.shape[0], 1),
            generator=generator,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        cumulative = probabilities.cumsum(dim=-1)
        sampled = (cumulative < uniform).sum(dim=-1).clamp(max=probabilities.shape[1] - 1)
        # Floating-point rounding in the cumulative sum can land on a masked node.
        feasible = action_mask.gather(1, sampled[:, None]).squeeze(1)
        return torch.where(feasible, sampled, greedy)


class _RolloutState:
    """Mutable per-step decoding state: visited mask, remaining capacity, cost, and log-probs."""

    def __init__(
        self,
        *,
        coords: Tensor,
        demands: Tensor,
        capacity: Tensor,
        depot_index: Tensor,
        node_mask: Tensor,
        local_adjacency: Tensor | None,
    ) -> None:
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.depot_index = depot_index
        self.node_mask = node_mask
        self.local_adjacency = local_adjacency

        batch_size, n_nodes, _ = coords.shape
        self.device = coords.device
        self.dtype = coords.dtype
        self.rows = torch.arange(batch_size, device=self.device)
        self.is_depot = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=self.device)
        self.is_depot[self.rows, depot_index] = True
        self.is_customer = node_mask & ~self.is_depot

        self.visited = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=self.device)
        self.current_node = depot_index.clone()
        self.remaining_capacity = capacity.clone()
        self.finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.cost = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.log_probability = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.entropy_total = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.step_counts = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        self.max_steps = 2 * int(self.is_customer.sum(dim=1).max().item())

    @classmethod
    def create(cls, encoding: PolicyEncoding, embedding_dim: int) -> _RolloutState:
        node_embeddings = encoding.node_embeddings
        if node_embeddings.ndim != 3:
            raise ValueError(
                "node_embeddings must have shape [batch, n_nodes, embedding_dim], got "
                f"{tuple(node_embeddings.shape)}"
            )
        if node_embeddings.shape[-1] != embedding_dim:
            raise ValueError(
                f"expected embedding_dim {embedding_dim}, got {node_embeddings.shape[-1]}"
            )
        dtype = node_embeddings.dtype
        node_mask = encoding.node_mask.to(dtype=torch.bool)
        state = cls(
            coords=encoding.coords.to(dtype=dtype),
            demands=encoding.demands.to(dtype=dtype),
            capacity=encoding.capacity.to(dtype=dtype),
            depot_index=encoding.depot_index.to(dtype=torch.long),
            node_mask=node_mask,
            local_adjacency=(
                None
                if encoding.local_adjacency is None
                else encoding.local_adjacency.to(dtype=torch.bool)
            ),
        )
        if not bool(state.is_customer.any(dim=1).all()):
            raise ValueError("every instance must contain at least one customer")
        oversized = state.demands > state.capacity[:, None] + _CAPACITY_TOLERANCE
        if bool((oversized & state.is_customer).any()):
            raise ValueError("every customer demand must fit within the vehicle capacity")
        return state

    def capacity_ratio(self) -> Tensor:
        return self.remaining_capacity / self.capacity.clamp_min(1e-8)

    def action_mask(self) -> Tensor:
        """Feasible next nodes: unserved customers that fit, plus the depot when away from it."""
        unvisited = self.is_customer & ~self.visited
        fits_capacity = self.demands <= self.remaining_capacity[:, None] + _CAPACITY_TOLERANCE
        away_from_depot = self.current_node != self.depot_index
        mask = (unvisited & fits_capacity) | (self.is_depot & away_from_depot[:, None])
        # Once every customer is served the vehicle idles at the depot; keep exactly one legal
        # action so the softmax is never fully masked.
        return mask | (self.is_depot & ~mask.any(dim=1, keepdim=True))

    def local_available(self, action_mask: Tensor) -> Tensor:
        """Restrict the action mask to the ``M_hat`` neighbourhood of the current node."""
        if self.local_adjacency is None:
            return action_mask
        neighbourhood = self.local_adjacency[self.rows, self.current_node]
        restricted = action_mask & neighbourhood
        # An exhausted neighbourhood would leave the local glimpse fully masked.
        return torch.where(restricted.any(dim=1, keepdim=True), restricted, action_mask)

    def step(
        self,
        action: Tensor,
        step_log_probability: Tensor,
        step_entropy: Tensor,
    ) -> Tensor:
        """Apply one action, returning the effective action (the depot for finished instances)."""
        active = ~self.finished
        active_values = active.to(dtype=self.dtype)
        effective = torch.where(active, action, self.depot_index)

        travelled = torch.linalg.norm(
            self.coords[self.rows, effective] - self.coords[self.rows, self.current_node],
            dim=-1,
        )
        self.cost = self.cost + travelled * active_values
        self.log_probability = self.log_probability + step_log_probability * active_values
        self.entropy_total = self.entropy_total + step_entropy * active_values
        self.step_counts = self.step_counts + active_values

        selected = torch.zeros_like(self.visited)
        selected.scatter_(1, effective[:, None], True)
        self.visited = self.visited | (selected & self.is_customer & active[:, None])

        selected_is_depot = self.is_depot.gather(1, effective[:, None]).squeeze(1)
        selected_demand = self.demands.gather(1, effective[:, None]).squeeze(1)
        served = self.remaining_capacity - selected_demand
        reloaded = torch.where(selected_is_depot, self.capacity, served)
        self.remaining_capacity = torch.where(active, reloaded, self.remaining_capacity)
        self.current_node = torch.where(active, effective, self.current_node)

        all_served = ~(self.is_customer & ~self.visited).any(dim=1)
        self.finished = self.finished | (all_served & (self.current_node == self.depot_index))
        return effective

    def final_cost(self) -> Tensor:
        """Add the closing depot leg; it is zero once the rollout ends at the depot."""
        closing: Tensor = torch.linalg.norm(
            self.coords[self.rows, self.current_node] - self.coords[self.rows, self.depot_index],
            dim=-1,
        )
        return self.cost + closing

    def mean_entropy(self) -> Tensor:
        return self.entropy_total / self.step_counts.clamp_min(1.0)


class CVRPPolicy(nn.Module):
    """Mask-guided encoder plus dual-pointer decoder (CMD Fig. 4b-c).

    The policy owns the trainable half of the pipeline; the diffusion prior stays frozen upstream
    and enters only through ``m_hat``. Setting ``use_local_encoder=False`` drops both the masked
    local encoder and the fusion block, which is the "no diffusion prior" encoder ablation.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        global_num_layers: int = 3,
        global_num_heads: int = 8,
        local_num_layers: int = 2,
        local_num_heads: int = 8,
        feed_forward_dim: int = 512,
        dropout: float = 0.0,
        decoder_num_heads: int = 8,
        clip_constant: float = 10.0,
        adjacency_mode: AdjacencyMode = "hard",
        local_threshold: float = 0.5,
        use_local_encoder: bool = True,
        use_local_pointer: bool = True,
        use_global_pointer: bool = True,
        use_savings_bias: bool = True,
        use_context_perception: bool = True,
        savings_weight: float = 1.0,
        savings_epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        if not 0.0 <= local_threshold <= 1.0:
            raise ValueError(f"local_threshold must be in [0, 1], got {local_threshold}")
        self.embedding_dim = embedding_dim
        self.use_local_encoder = use_local_encoder
        self.use_local_pointer = use_local_pointer
        self.local_threshold = local_threshold

        self.global_encoder = GlobalEncoder(
            embedding_dim=embedding_dim,
            num_layers=global_num_layers,
            num_heads=global_num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
        )
        self.local_encoder = (
            LocalMaskedEncoder(
                embedding_dim=embedding_dim,
                num_layers=local_num_layers,
                num_heads=local_num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout,
                adjacency_mode=adjacency_mode,
                hard_threshold=max(local_threshold, 1e-6),
            )
            if use_local_encoder
            else None
        )
        self.fusion_encoder = (
            FusionEncoder(
                embedding_dim=embedding_dim,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout,
            )
            if use_local_encoder
            else None
        )
        self.decoder = DualPointerDecoder(
            embedding_dim=embedding_dim,
            num_heads=decoder_num_heads,
            clip_constant=clip_constant,
            use_local_pointer=use_local_pointer,
            use_global_pointer=use_global_pointer,
            use_savings_bias=use_savings_bias,
            use_context_perception=use_context_perception,
            savings_weight=savings_weight,
            savings_epsilon=savings_epsilon,
        )

    def encode(
        self,
        coords: Tensor,
        demands: Tensor,
        capacity: Tensor,
        depot_index: Tensor,
        node_mask: Tensor,
        *,
        m_hat: Tensor | None = None,
        customer_node_indices: Tensor | None = None,
        customer_mask: Tensor | None = None,
    ) -> PolicyEncoding:
        """Encode one padded CVRP batch into reusable decoder inputs."""
        has_prior = (
            m_hat is not None and customer_node_indices is not None and customer_mask is not None
        )
        if self.local_encoder is not None and not has_prior:
            raise ValueError(
                "use_local_encoder=True requires m_hat, customer_node_indices, and customer_mask"
            )

        global_output: GlobalEncoderOutput = self.global_encoder(
            coords, demands, capacity, depot_index, node_mask
        )
        node_embeddings = global_output.node_embeddings
        graph_embedding = global_output.graph_embedding
        local_adjacency: Tensor | None = None
        fusion_gate: Tensor | None = None

        if has_prior and (self.local_encoder is not None or self.use_local_pointer):
            assert m_hat is not None
            assert customer_node_indices is not None
            assert customer_mask is not None
            local_adjacency = build_decoder_local_adjacency(
                m_hat,
                customer_node_indices,
                customer_mask,
                depot_index,
                node_mask,
                threshold=self.local_threshold,
            )
            if self.local_encoder is not None and self.fusion_encoder is not None:
                local_output: LocalMaskedEncoderOutput = self.local_encoder(
                    node_embeddings,
                    m_hat,
                    customer_node_indices,
                    customer_mask,
                    depot_index,
                    node_mask,
                )
                fused: FusionEncoderOutput = self.fusion_encoder(
                    node_embeddings,
                    local_output.node_embeddings,
                    node_mask,
                )
                node_embeddings = fused.node_embeddings
                graph_embedding = fused.graph_embedding
                fusion_gate = fused.fusion_gate

        return PolicyEncoding(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
            coords=coords,
            demands=demands,
            capacity=capacity,
            depot_index=depot_index,
            node_mask=node_mask.to(dtype=torch.bool),
            local_adjacency=local_adjacency if self.use_local_pointer else None,
            fusion_gate=fusion_gate,
        )

    def rollout(
        self,
        encoding: PolicyEncoding,
        *,
        decode_mode: DecodeMode = "greedy",
        num_starts: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DecoderRollout:
        """Decode ``num_starts`` solutions per instance, flattened along the batch dimension.

        ``num_starts=None`` applies CMD Algorithm 1 ``NStart`` via :func:`select_nstart_nodes`.
        With more than one start the returned tensors have batch size ``batch * num_starts``, laid
        out so that ``tensor.view(batch, num_starts, ...)`` groups the starts of one instance.
        """
        start_table: Tensor | None
        if num_starts is None:
            start_table = select_nstart_nodes(
                encoding.coords, encoding.depot_index, encoding.node_mask
            )
            num_starts = start_table.shape[1]
        elif num_starts < 1:
            raise ValueError(f"num_starts must be >= 1, got {num_starts}")
        else:
            start_table = (
                None
                if num_starts == 1
                else select_start_nodes(
                    encoding.coords,
                    encoding.depot_index,
                    encoding.node_mask,
                    num_starts,
                )
            )
        if start_table is None:
            single: DecoderRollout = self.decoder(
                encoding, decode_mode=decode_mode, generator=generator
            )
            return single

        encoded = encoding if num_starts == 1 else repeat_encoding(encoding, num_starts)
        multi: DecoderRollout = self.decoder(
            encoded,
            decode_mode=decode_mode,
            generator=generator,
            start_nodes=start_table.reshape(-1),
        )
        return multi


def repeat_encoding(encoding: PolicyEncoding, num_starts: int) -> PolicyEncoding:
    """Tile every per-instance tensor ``num_starts`` times, grouping starts by instance."""

    def tile(tensor: Tensor) -> Tensor:
        return tensor.repeat_interleave(num_starts, dim=0)

    return PolicyEncoding(
        node_embeddings=tile(encoding.node_embeddings),
        graph_embedding=tile(encoding.graph_embedding),
        coords=tile(encoding.coords),
        demands=tile(encoding.demands),
        capacity=tile(encoding.capacity),
        depot_index=tile(encoding.depot_index),
        node_mask=tile(encoding.node_mask),
        local_adjacency=(
            None if encoding.local_adjacency is None else tile(encoding.local_adjacency)
        ),
        fusion_gate=(None if encoding.fusion_gate is None else tile(encoding.fusion_gate)),
    )
