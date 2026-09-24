"""Neural model components for CMD-based CVRP solving."""

from vrp_diffusion_quantum.models.fusion_encoder import FusionEncoder, FusionEncoderOutput
from vrp_diffusion_quantum.models.global_encoder import (
    GlobalEncoder,
    GlobalEncoderOutput,
    build_global_node_features,
)
from vrp_diffusion_quantum.models.local_masked_encoder import (
    LocalMaskedEncoder,
    LocalMaskedEncoderOutput,
    build_local_attention_prior,
)
from vrp_diffusion_quantum.models.paper_cmd_encoder import (
    PaperGlobalGATEncoder,
    PaperMaskedGATEncoder,
    PaperSumMLPFusion,
    load_diffusion_gat_checkpoint,
)

__all__ = [
    "FusionEncoder",
    "FusionEncoderOutput",
    "GlobalEncoder",
    "GlobalEncoderOutput",
    "LocalMaskedEncoder",
    "LocalMaskedEncoderOutput",
    "PaperGlobalGATEncoder",
    "PaperMaskedGATEncoder",
    "PaperSumMLPFusion",
    "build_global_node_features",
    "build_local_attention_prior",
    "load_diffusion_gat_checkpoint",
]
