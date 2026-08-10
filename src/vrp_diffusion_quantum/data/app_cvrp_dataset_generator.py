"""CVRP Dataset Generator — Streamlit UI for generate + solve.

Run:
    streamlit run src/vrp_diffusion_quantum/data/app_cvrp_dataset_generator.py
"""

from __future__ import annotations

import importlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# .../src/vrp_diffusion_quantum/data/<this file> -> repo root is parents[3]
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Streamlit can keep a stale generate_cvrp module in sys.modules across reruns.
import vrp_diffusion_quantum.data.generate_cvrp as _gen_mod  # noqa: E402

_gen_mod = importlib.reload(_gen_mod)

ROUTE_SIZE_RANGES = _gen_mod.ROUTE_SIZE_RANGES
CVRPDataset = _gen_mod.CVRPDataset
CustomerMode = _gen_mod.CustomerMode
DemandMode = _gen_mod.DemandMode
DepotMode = _gen_mod.DepotMode
demand_bounds_for_mode = _gen_mod.demand_bounds_for_mode
generate_dataset_from_config = _gen_mod.generate_dataset_from_config
load_dataset = _gen_mod.load_dataset
save_dataset = _gen_mod.save_dataset
if hasattr(_gen_mod, "suggested_capacity"):
    suggested_capacity = _gen_mod.suggested_capacity
else:
    # Fallback if an old module object is still cached.
    def suggested_capacity(n_customers: int, high_demand: int) -> int:
        n = int(n_customers)
        if n <= 20:
            floor = max(1, round(30 * n / 20)) if n < 20 else 30
        elif n <= 50:
            floor = round(30 + 10 * (n - 20) / 30)
        elif n <= 100:
            floor = round(40 + 10 * (n - 50) / 50)
        else:
            floor = round(50 + 0.2 * (n - 100))
        return max(int(high_demand), floor)


from vrp_diffusion_quantum.data.export_examples import (  # noqa: E402
    _discover_labeled_sizes as _labeled_sizes,
)
from vrp_diffusion_quantum.data.export_examples import (  # noqa: E402
    export_run,
)
from vrp_diffusion_quantum.data.solve_cvrp import (  # noqa: E402
    CVRPSolution,
    FleetMode,
    save_labels,
    solve_dataset,
)

DEFAULT_DATA = ROOT / "data" / "raw" / "cvrp"
DEFAULT_GEN_CFG = ROOT / "configs" / "data" / "cvrp.yaml"
DEFAULT_SOLVE_CFG = ROOT / "configs" / "data" / "solve_labels.yaml"
STEM_RE = re.compile(r"^cvrp(\d+)$")
CLUSTER_MODES = {"clustered", "random_clustered"}

ROUTE_COLORS = [
    "#5EEAD4",
    "#FBBF24",
    "#60A5FA",
    "#F472B6",
    "#34D399",
    "#FB923C",
    "#A3E635",
    "#38BDF8",
]

st.set_page_config(
    page_title="CVRP Dataset Generator",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet">
    <style>
      :root {
        --bg0: #080a0c;
        --bg1: #0b0d10;
        --surface: #151920;
        --ink: #e8eaed;
        --muted: #8b93a0;
        --line: #2a313c;
        --accent: #5eead4;
        --accent-ink: #04120f;
      }
      html, body, [class*="css"] { font-family: "DM Sans", sans-serif; color: var(--ink); }
      .stApp {
        background:
          radial-gradient(ellipse 70% 45% at 50% -15%, rgba(94,234,212,0.10), transparent 55%),
          linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
      }
      #MainMenu { visibility: hidden !important; }
      header[data-testid="stHeader"] { display: none !important; }
      [data-testid="stToolbar"] { display: none !important; }
      [data-testid="stDecoration"] { display: none !important; }
      [data-testid="stStatusWidget"] { display: none !important; }
      .stAppDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
      [data-testid="stSidebar"] { display: none !important; }
      [data-testid="stSidebarCollapsedControl"] { display: none !important; }
      footer { visibility: hidden !important; }
      .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 760px !important;
      }
      h1, h2, h3 {
        font-family: "Sora", sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.04em;
      }
      .brand { text-align: center; margin: 0 0 1.5rem; }
      .brand-mark {
        font-family: "Sora", sans-serif;
        font-weight: 700;
        font-size: 1.85rem;
        letter-spacing: -0.045em;
        color: var(--ink);
        line-height: 1.15;
      }
      .brand-mark span { color: var(--accent); }
      .section-label {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.25rem 0 0.35rem;
      }
      .blurb {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
        margin: 0 0 0.8rem;
      }
      .path-chip {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.78rem;
        color: var(--accent);
        background: rgba(94,234,212,0.08);
        border: 1px solid rgba(94,234,212,0.25);
        border-radius: 6px;
        padding: 0.45rem 0.65rem;
        margin: 0.35rem 0 0.8rem;
        word-break: break-all;
      }
      label, [data-testid="stWidgetLabel"] p { color: var(--muted) !important; }
      div[data-testid="stRadio"] > label { display: none; }
      div[data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.4rem; justify-content: center;
      }
      div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        min-width: 8.5rem;
        justify-content: center;
      }
      div[data-testid="stRadio"] label[data-baseweb="radio"] p { color: var(--ink) !important; }
      div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        border-color: var(--accent);
        background: rgba(94,234,212,0.12);
      }
      div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: var(--accent) !important;
      }
      .stButton > button[kind="primary"],
      .stButton > button[data-testid="baseButton-primary"] {
        background: var(--accent) !important;
        border: none !important;
        color: var(--accent-ink) !important;
        border-radius: 8px !important;
        font-family: "Sora", sans-serif !important;
        font-weight: 600 !important;
        height: 2.75rem;
      }
      code, pre { font-family: "JetBrains Mono", monospace !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _blurb(text: str) -> None:
    st.markdown(f'<p class="blurb">{text}</p>', unsafe_allow_html=True)


def _path_chip(path: Path) -> None:
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    st.markdown(f'<div class="path-chip">{shown}</div>', unsafe_allow_html=True)


def _short_demand(mode: str) -> str:
    mapping = {
        "uniform": "uniform",
        "unitary": "unitary",
        "small_demand_wide_spread": "sd-wide",
        "small_demand_narrow_spread": "sd-narrow",
        "large_demand_wide_spread": "ld-wide",
        "large_demand_narrow_spread": "ld-narrow",
        "quadrant": "quadrant",
        "many_small_few_large": "msfl",
        "custom": "custom",
    }
    return mapping.get(mode, mode.replace("_", "-")[:24])


def dataset_run_name(config: dict[str, Any], *, stamped: bool = False) -> str:
    """Build a filesystem-safe folder name describing the generate config.

    When ``stamped`` is True, prefixes ``YYYYMMDD_HHMMSS_`` so each run gets a unique
    folder even with the same configuration.
    """
    sizes = "-".join(str(s) for s in config["sizes"])
    parts = [
        f"s{config['seed']}",
        f"n{sizes}",
        f"x{config['num_instances']}",
        f"dep-{config['depot_mode']}",
        f"cust-{config['customer_mode']}",
        f"dem-{_short_demand(str(config['demand_mode']))}",
    ]
    if config.get("random_demand_bounds"):
        parts.append("dbrnd")
    else:
        by_n = config.get("demand_bounds_by_n") or {}
        if by_n:
            bits = []
            for size in config["sizes"]:
                bounds = by_n.get(str(size)) or by_n.get(size)
                if bounds:
                    bits.append(f"{size}:{bounds['low']}-{bounds['high']}")
            if bits:
                parts.append("d" + ",".join(bits))
    if config.get("route_size") is not None:
        rs = str(config["route_size"]).replace(" ", "")
        parts.append(f"rs-{rs}")
    elif config.get("random_capacity"):
        caps = config.get("capacity_max_by_n") or {}
        tag = "capcurve" if config.get("capacity_curve") else "caprnd"
        if caps:
            bits = [f"{s}:{caps.get(str(s), caps.get(s))}" for s in config["sizes"]]
            parts.append(f"{tag}-" + ",".join(bits))
        else:
            parts.append(tag)
    elif config.get("capacity_by_n"):
        caps = config["capacity_by_n"]
        bits = [f"{s}:{caps.get(str(s), caps.get(s))}" for s in config["sizes"]]
        parts.append("cap" + ",".join(bits))
    else:
        parts.append("cap-default")
    if config["customer_mode"] in CLUSTER_MODES:
        parts.append(f"decay{config['cluster_decay']}")
    name = "_".join(parts)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    if stamped:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{name}"
    return name


def _cleanup_empty_run_folders(root: Path) -> int:
    """Remove empty / incomplete run folders left behind by failed generates."""
    if not root.is_dir():
        return 0
    removed = 0
    for child in list(root.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name == "labels":
            continue
        # Keep folders that already have dataset CSVs.
        if any(child.glob("cvrp*_nodes.csv")):
            continue
        # Incomplete: no CSVs (maybe only config.yaml or totally empty).
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed


def _discover_run_folders(root: Path) -> list[str]:
    """Return relative folder names under ``root`` that contain CVRP CSV pairs."""
    if not root.is_dir():
        return []
    _cleanup_empty_run_folders(root)
    found: list[str] = []
    # Nested run folders (preferred). Newest timestamped runs first.
    for child in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not child.is_dir() or child.name.startswith(".") or child.name == "labels":
            continue
        if any(child.glob("cvrp*_nodes.csv")):
            found.append(child.name)
    # Legacy flat layout: CSVs directly in root.
    if any(root.glob("cvrp*_nodes.csv")):
        found.insert(0, ".")
    return found


def _discover_sizes(dataset_dir: Path) -> list[int]:
    """Return customer sizes found on disk, sorted ascending (20 before 100)."""
    if not dataset_dir.is_dir():
        return []
    sizes: list[int] = []
    for nodes_path in dataset_dir.glob("cvrp*_nodes.csv"):
        stem = nodes_path.name[: -len("_nodes.csv")]
        match = STEM_RE.match(stem)
        if match is None:
            continue
        if (dataset_dir / f"{stem}_instances.csv").is_file():
            sizes.append(int(match.group(1)))
    return sorted(sizes)


def _style_axes(ax: Axes, title: str) -> None:
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.grid(True, color="#2a313c", alpha=0.85, lw=0.55)
    ax.set_title(title, fontsize=11, color="#e8eaed", fontweight="600", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2a313c")


def _plot_instance(coords: np.ndarray, demands: np.ndarray, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fig.patch.set_facecolor("#0b0d10")
    ax.set_facecolor("#151920")
    depot = coords[0]
    cust = coords[1:]
    sizes = 34 + 70 * (demands[1:] / max(float(demands[1:].max()), 1e-9))
    ax.scatter(
        cust[:, 0], cust[:, 1], s=sizes, c="#c5cad3", alpha=0.9, edgecolors="#0b0d10", lw=0.5
    )
    ax.scatter(
        [depot[0]],
        [depot[1]],
        s=150,
        c="#5eead4",
        marker="s",
        edgecolors="#0b0d10",
        lw=1.0,
        zorder=5,
    )
    _style_axes(ax, title)
    fig.tight_layout()
    return fig


def _plot_solution(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    solution: CVRPSolution,
    title: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    fig.patch.set_facecolor("#0b0d10")
    ax.set_facecolor("#151920")
    depot = coords[0]
    cust = coords[1:]
    dem_cust = demands[1:]
    dem_max = max(float(dem_cust.max()), 1e-9)
    sizes = 26 + 55 * (dem_cust / dem_max)
    ax.scatter(
        cust[:, 0],
        cust[:, 1],
        s=sizes,
        c="#8b93a0",
        alpha=0.75,
        edgecolors="#0b0d10",
        lw=0.4,
        zorder=3,
    )
    ax.scatter(
        [depot[0]],
        [depot[1]],
        s=160,
        c="#e8eaed",
        marker="s",
        edgecolors="#0b0d10",
        lw=1.1,
        zorder=6,
    )
    # Depot capacity label.
    ax.annotate(
        f"cap={capacity:g}",
        (depot[0], depot[1]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=8,
        color="#5eead4",
        zorder=7,
    )
    for i, route in enumerate(solution.routes):
        if not route:
            continue
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        path = [depot] + [coords[int(n)] for n in route] + [depot]
        ax.plot(
            [p[0] for p in path],
            [p[1] for p in path],
            color=color,
            lw=1.7,
            alpha=0.92,
            zorder=4,
        )
        load = float(sum(float(demands[int(n)]) for n in route))
        ax.scatter(
            [coords[int(n)][0] for n in route],
            [coords[int(n)][1] for n in route],
            s=40,
            c=color,
            edgecolors="#0b0d10",
            lw=0.55,
            zorder=5,
            label=f"R{i} load={load:.3g}/{capacity:g}",
        )
        for n in route:
            node = int(n)
            ax.annotate(
                f"{float(demands[node]):.3g}",
                (coords[node, 0], coords[node, 1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#e8eaed",
                alpha=0.9,
                zorder=7,
            )
    _style_axes(ax, title)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=False,
        labelcolor="#c5cad3",
    )
    fig.tight_layout()
    return fig


_CURVE_EDITOR = components.declare_component(
    "cvrp_capacity_curve",
    path=str(Path(__file__).resolve().parent / "capacity_curve_editor"),
)


def _normalize_knots(knots: list[tuple[int, float]], lo: int, hi: int) -> list[tuple[int, float]]:
    """Pin endpoints, clamp weights, sort by capacity, drop duplicate x."""
    lo_i, hi_i = int(lo), int(hi)
    if not knots:
        return [(lo_i, 0.5), (hi_i, 0.5)]
    cleaned = [
        (round(float(np.clip(x, lo_i, hi_i))), float(np.clip(w, 0.0, 1.0))) for x, w in knots
    ]
    cleaned.sort(key=lambda t: t[0])
    merged: list[tuple[int, float]] = []
    for x, w in cleaned:
        if merged and merged[-1][0] == x:
            merged[-1] = (x, max(merged[-1][1], w))
        else:
            merged.append((x, w))
    if merged[0][0] != lo_i:
        merged.insert(0, (lo_i, merged[0][1]))
    else:
        merged[0] = (lo_i, merged[0][1])
    if merged[-1][0] != hi_i:
        merged.append((hi_i, merged[-1][1]))
    else:
        merged[-1] = (hi_i, merged[-1][1])
    return merged


def _shape_preset_knots(
    lo: int, hi: int, shape: str, peak: int | None = None, n_points: int = 7
) -> list[tuple[int, float]]:
    """Build smooth preset weight knots on ``[lo, hi]``."""
    lo_i, hi_i = int(lo), int(hi)
    if hi_i <= lo_i:
        return [(lo_i, 1.0)]
    xs = np.linspace(lo_i, hi_i, n_points)
    t = (xs - lo_i) / float(hi_i - lo_i)
    peak_t = 0.5
    if peak is not None:
        peak_t = float(np.clip((peak - lo_i) / float(hi_i - lo_i), 0.0, 1.0))

    if shape == "flat":
        ws = np.ones_like(t)
    elif shape == "low":
        ws = (1.0 - t) ** 1.6
    elif shape == "high":
        ws = t**1.6
    elif shape == "edges":
        ws = 0.15 + 0.85 * (2.0 * (t - 0.5)) ** 2
    else:  # bell
        sigma = 0.18
        ws = np.exp(-0.5 * ((t - peak_t) / sigma) ** 2)

    ws = ws / float(ws.max()) if float(ws.max()) > 0 else np.ones_like(ws)
    return _normalize_knots(
        [(round(float(x)), float(w)) for x, w in zip(xs, ws, strict=True)],
        lo_i,
        hi_i,
    )


def _edit_capacity_curve(size: int, lo: int, hi: int, peak: int) -> list[tuple[int, float]]:
    """Modern drag-and-drop probability curve editor with shape presets."""
    state_key = f"cap_knots_{size}_{lo}_{hi}"
    rev_key = f"{state_key}_rev"
    if state_key not in st.session_state:
        st.session_state[state_key] = _shape_preset_knots(lo, hi, "bell", peak)
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0

    st.caption(f"CVRP-{size} · high demand {lo} → capacity max {hi}")
    presets = st.columns(5)
    preset_map = {
        0: ("Flat", "flat"),
        1: ("Bell", "bell"),
        2: ("Prefer low", "low"),
        3: ("Prefer high", "high"),
        4: ("U-shape", "edges"),
    }
    for i, col in enumerate(presets):
        label, shape = preset_map[i]
        with col:
            if st.button(label, key=f"preset_{state_key}_{shape}", use_container_width=True):
                st.session_state[state_key] = _shape_preset_knots(lo, hi, shape, peak)
                st.session_state[rev_key] = int(st.session_state[rev_key]) + 1
                st.rerun()

    knots = _normalize_knots(list(st.session_state[state_key]), int(lo), int(hi))
    st.session_state[state_key] = knots
    payload = [{"x": int(x), "y": float(y)} for x, y in knots]

    result = _CURVE_EDITOR(
        knots=payload,
        lo=int(lo),
        hi=int(hi),
        revision=int(st.session_state[rev_key]),
        key=f"editor_{state_key}_{st.session_state[rev_key]}",
        default=payload,
    )
    if result is not None:
        parsed = _normalize_knots(
            [(int(p["x"]), float(p["y"])) for p in result],
            int(lo),
            int(hi),
        )
        if parsed != knots:
            st.session_state[state_key] = parsed

    return list(st.session_state[state_key])


def _render_generate(defaults: dict[str, Any], data_root: Path) -> None:
    st.markdown('<div class="section-label">Generate</div>', unsafe_allow_html=True)
    _blurb(
        "Sample CVRP instances into a config-named folder under the workspace. "
        "CSVs store raw integer demands and the real vehicle capacity."
    )

    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input(
            "seed",
            min_value=0,
            value=int(defaults.get("seed", 42)),
            step=1,
            help="Base RNG seed. Each size gets a derived seed.",
        )
    with c2:
        num_instances = st.number_input(
            "instances / size",
            min_value=1,
            max_value=100_000,
            value=min(int(defaults.get("num_instances", 100)), 32),
            step=1,
            help="Problems per customer count. Keep small in the UI.",
        )
    default_sizes = [int(s) for s in defaults.get("sizes", [20, 50, 100])]
    if "size_options" not in st.session_state:
        st.session_state.size_options = sorted(set([*default_sizes, 20, 50, 100, 200]))
    if "gen_sizes_multi" not in st.session_state:
        st.session_state.gen_sizes_multi = list(default_sizes)

    def _on_sizes_change() -> None:
        """Coerce typed sizes to ints (safe inside widget callback)."""
        raw = list(st.session_state.get("gen_sizes_multi", []))
        coerced: list[int] = []
        for item in raw:
            try:
                n = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 100_000:
                coerced.append(n)
        coerced = sorted(set(coerced))
        st.session_state.gen_sizes_multi = coerced
        st.session_state.size_options = sorted(
            {int(x) for x in st.session_state.get("size_options", [])} | set(coerced)
        )

    sizes_raw = st.multiselect(
        "sizes (customers)",
        options=st.session_state.size_options,
        key="gen_sizes_multi",
        accept_new_options=True,
        on_change=_on_sizes_change,
        placeholder="Pick tags or type a size and press Enter",
        help="Tag chips for each customer count. Type any new size (e.g. 75) and press Enter.",
    )
    sizes: list[int] = []
    for raw in sizes_raw:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            st.error(f"Size must be an integer, got {raw!r}.")
            return
        if n < 1 or n > 100_000:
            st.error(f"Size must be between 1 and 100000, got {n}.")
            return
        sizes.append(n)
    sizes = sorted(set(sizes))

    st.markdown('<div class="section-label">Placement</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        depot_mode = st.selectbox(
            "depot",
            options=list(get_args(DepotMode)),
            index=0,
            help="Depot location style in the unit square.",
        )
    with c2:
        customer_mode = st.selectbox(
            "customers",
            options=list(get_args(CustomerMode)),
            index=0,
            help="Customer layout: random, clustered, or mixed.",
        )
    cluster_decay = float(defaults.get("cluster_decay", 0.04))
    if customer_mode in CLUSTER_MODES:
        cluster_decay = st.slider(
            "cluster decay",
            0.01,
            0.20,
            float(defaults.get("cluster_decay", 0.04)),
            0.01,
            help="Smaller = tighter clusters. Only used for clustered layouts.",
        )

    if not sizes:
        st.warning("Pick at least one size.")
        return

    st.markdown('<div class="section-label">Demand</div>', unsafe_allow_html=True)
    _blurb(
        "Preset modes fill low/high per size (dimmed). "
        "Use <b>custom</b> to set bounds <b>for each size</b>, or randomize bounds per instance."
    )
    demand_mode = st.selectbox(
        "demand mode",
        options=list(get_args(DemandMode)),
        index=0,
        help="Preset Uchoa / classic regimes, or custom [low, high] per size.",
    )
    random_demand_bounds = st.toggle(
        "random low/high demand per instance (1..100)",
        value=False,
        help="Each instance draws its own [low, high] with 1 <= low <= high <= 100.",
    )

    preset_locked = demand_mode != "custom" and not random_demand_bounds
    if demand_mode == "custom" or random_demand_bounds:
        default_low, default_high = 1, 10
    else:
        default_low, default_high = demand_bounds_for_mode(demand_mode)

    demand_bounds_by_n: dict[int, dict[str, int | None]] = {}
    if random_demand_bounds:
        st.caption("Per-size low/high not used — each instance samples its own bounds.")
        for size in sizes:
            demand_bounds_by_n[int(size)] = {"low": None, "high": None}
    else:
        st.markdown("**Low / high demand per size**")
        hdr1, hdr2, hdr3 = st.columns([1.1, 1, 1])
        with hdr1:
            st.caption("size")
        with hdr2:
            st.caption("low demand")
        with hdr3:
            st.caption("high demand")
        for size in sizes:
            c1, c2, c3 = st.columns([1.1, 1, 1])
            with c1:
                st.markdown(f"CVRP-{size}")
            with c2:
                low_v = int(
                    st.number_input(
                        f"low CVRP-{size}",
                        min_value=1,
                        max_value=100,
                        value=int(default_low),
                        step=1,
                        disabled=preset_locked,
                        key=f"demand_low_{size}_{demand_mode}",
                        label_visibility="collapsed",
                        help=f"Low demand for CVRP-{size}",
                    )
                )
            with c3:
                high_v = int(
                    st.number_input(
                        f"high CVRP-{size}",
                        min_value=1,
                        max_value=100,
                        value=int(default_high),
                        step=1,
                        disabled=preset_locked,
                        key=f"demand_high_{size}_{demand_mode}",
                        label_visibility="collapsed",
                        help=f"High demand for CVRP-{size}",
                    )
                )
            if low_v > high_v:
                st.error(f"CVRP-{size}: low demand cannot exceed high demand.")
                return
            demand_bounds_by_n[int(size)] = {"low": low_v, "high": high_v}
        if preset_locked:
            st.caption("Low/high filled from the selected demand mode (read-only).")

    st.markdown('<div class="section-label">Capacity</div>', unsafe_allow_html=True)
    capacity_mode = st.radio(
        "capacity mode",
        options=["fixed", "random", "route_size"],
        horizontal=True,
        format_func=lambda m: {
            "fixed": "Fixed",
            "random": "Random",
            "route_size": "From route_size",
        }[m],
        help="Fixed / random defaults follow max(high demand, size floor).",
    )

    route_size: float | str | None = None
    random_capacity = False
    capacity_curve = False
    capacity_by_n: dict[int, int | None] = {int(s): None for s in sizes}
    capacity_max_by_n: dict[int, int | None] = {int(s): None for s in sizes}
    capacity_weights_by_n: dict[int, list[tuple[int, float]] | None] = {int(s): None for s in sizes}

    if capacity_mode == "route_size":
        route_preset = st.selectbox(
            "route_size",
            options=["(custom)", *sorted(ROUTE_SIZE_RANGES)],
            index=1,
        )
        if route_preset == "(custom)":
            route_size = float(st.number_input("custom route_size", value=8.0, min_value=1.0))
        else:
            route_size = route_preset

        # Estimated capacity is per-instance and stochastic:
        #   capacity = max(round(r * mean_demand), max(max_demand, high_floor))
        # r is fixed (custom) or sampled per instance (preset range); mean_demand
        # varies per instance. So we can only show an estimate, not an exact value.
        if isinstance(route_size, str):
            r_lo, r_hi = ROUTE_SIZE_RANGES[route_size]
            r_mid = (r_lo + r_hi) / 2.0
            r_label = f"{route_size} ≈ U({r_lo:g}, {r_hi:g}), midpoint {r_mid:g}"
        else:
            r_mid = float(route_size)
            r_label = f"{r_mid:g} (fixed)"

        st.markdown("**Estimated capacity per size**")
        for size in sizes:
            bounds = demand_bounds_by_n[int(size)]
            if random_demand_bounds:
                mean_est = 50.0  # rough E[demand] when bounds are random per instance
                high_floor = 100
                mean_note = "≈50 (random per-instance bounds)"
            else:
                low_v = int(bounds["low"] or 1)
                high_v = int(bounds["high"] or 1)
                mean_est = (low_v + high_v) / 2.0
                high_floor = high_v
                mean_note = f"({low_v}+{high_v})/2 = {mean_est:g}"
            raw = round(r_mid * mean_est)
            est_cap = max(raw, high_floor)
            # No widget key: a keyed disabled input caches its first value and never
            # refreshes when route_size / demand bounds change.
            st.number_input(
                f"CVRP-{size} est. capacity",
                value=int(est_cap),
                step=1,
                disabled=True,
                help="Estimate only — actual capacity is computed per instance at generation.",
            )
            st.caption(
                f"≈ max(round(r x mean demand), high floor) = "
                f"max(round({r_mid:g} x {mean_est:g}), {high_floor}) = "
                f"max({raw}, {high_floor}) = **{est_cap}**  ·  "
                f"r: {r_label}  ·  mean demand: {mean_note}. "
                "Varies per instance."
            )
    elif capacity_mode == "fixed":
        st.markdown("**Capacity per size**")
        for size in sizes:
            bounds = demand_bounds_by_n[int(size)]
            high_floor = 100 if random_demand_bounds else int(bounds["high"] or 1)
            suggested = suggested_capacity(int(size), high_floor)
            capacity_by_n[int(size)] = int(
                st.number_input(
                    f"CVRP-{size} capacity",
                    min_value=high_floor,
                    value=suggested,
                    step=1,
                    key=f"cap_fixed_{size}_{high_floor}",
                    help=f"Default {suggested} = max(high={high_floor}, size floor).",
                )
            )
    else:
        random_capacity = True
        dist_kind = st.radio(
            "random distribution",
            options=["uniform", "curve"],
            horizontal=True,
            format_func=lambda m: {
                "uniform": "Uniform",
                "curve": "Shaped curve",
            }[m],
            help=(
                "Uniform: equal chance from high demand → max. "
                "Shaped curve: drag handles on a live probability graph."
            ),
        )
        capacity_curve = dist_kind == "curve"
        st.markdown("**Capacity max per size** (floor = high demand)")
        for size in sizes:
            bounds = demand_bounds_by_n[int(size)]
            high_floor = 100 if random_demand_bounds else int(bounds["high"] or 1)
            suggested = suggested_capacity(int(size), high_floor)
            cap_max = int(
                st.number_input(
                    f"CVRP-{size} capacity max",
                    min_value=high_floor,
                    value=max(suggested * 2, high_floor + 20),
                    step=1,
                    key=f"cap_max_{size}_{high_floor}",
                    help=(
                        f"Sample in [{high_floor}, max]; default max is 2x suggested ({suggested})."
                    ),
                )
            )
            capacity_max_by_n[int(size)] = cap_max

        if capacity_curve:
            st.markdown('<div class="section-label">Shape</div>', unsafe_allow_html=True)
            _blurb(
                "Drag the yellow handles. Double-click to add a point, "
                "right-click a handle to remove it. "
                "Presets jump-start a shape — then tweak freely."
            )
            for size in sizes:
                bounds = demand_bounds_by_n[int(size)]
                high_floor = 100 if random_demand_bounds else int(bounds["high"] or 1)
                suggested = suggested_capacity(int(size), high_floor)
                cap_max = int(capacity_max_by_n[int(size)] or suggested)
                with st.expander(f"Curve — CVRP-{size}", expanded=len(sizes) == 1):
                    capacity_weights_by_n[int(size)] = _edit_capacity_curve(
                        int(size), high_floor, cap_max, suggested
                    )

    config_preview = {
        "seed": int(seed),
        "sizes": list(sizes),
        "num_instances": int(num_instances),
        "depot_mode": depot_mode,
        "customer_mode": customer_mode,
        "demand_mode": demand_mode,
        "demand_bounds_by_n": {str(k): v for k, v in demand_bounds_by_n.items()},
        "random_demand_bounds": bool(random_demand_bounds),
        "route_size": route_size,
        "capacity_by_n": {str(k): v for k, v in capacity_by_n.items()},
        "capacity_max_by_n": {str(k): v for k, v in capacity_max_by_n.items()},
        "capacity_weights_by_n": {
            str(k): (list(v) if v is not None else None) for k, v in capacity_weights_by_n.items()
        },
        "random_capacity": bool(random_capacity),
        "capacity_curve": bool(capacity_curve),
        "cluster_decay": float(cluster_decay),
    }
    config_slug = dataset_run_name(config_preview, stamped=False)

    st.markdown('<div class="section-label">Output folder</div>', unsafe_allow_html=True)
    _blurb(
        "Each Generate creates a new folder: "
        "<code>YYYYMMDD_HHMMSS_</code> + config name "
        "(same settings never overwrite a previous run)."
    )
    _path_chip(data_root / f"<timestamp>_{config_slug}")

    generate = st.button("Generate", type="primary", use_container_width=True)
    if not generate:
        return

    run_name = dataset_run_name(config_preview, stamped=True)
    out_dir = data_root / run_name

    # Generate fully in memory first so a bad config never leaves an empty folder.
    progress = st.progress(0, text="Generating…")
    generated: list[tuple[int, CVRPDataset]] = []
    previews: list[tuple[int, np.ndarray, np.ndarray]] = []
    try:
        for i, size in enumerate(sizes):
            progress.progress(i / max(len(sizes), 1), text=f"CVRP-{size}")
            # Single source of truth for config -> instances (shared with the exporter's
            # raw regeneration), so a re-export reproduces exactly this data.
            dataset = generate_dataset_from_config(config_preview, int(size))
            generated.append((int(size), dataset))
            previews.append((int(size), dataset.coords[0], dataset.demands[0]))
    except Exception as exc:
        progress.empty()
        st.error(f"Generate failed — no folder written. {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for size, dataset in generated:
            save_dataset(dataset, out_dir / f"cvrp{size}")
        snapshot = {
            **config_preview,
            "output_dir": str(out_dir.relative_to(ROOT)),
            "run_name": run_name,
        }
        (out_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
    except Exception as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        progress.empty()
        st.error(f"Save failed — removed incomplete folder. {exc}")
        return

    progress.progress(1.0, text="Done")
    st.success(f"Saved dataset run `{run_name}`")
    _path_chip(out_dir)

    st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
    _blurb("Instance 0 per size. Depot in teal; point size scales with demand.")
    cols = st.columns(len(previews))
    for col, (size, coords, demands) in zip(cols, previews, strict=True):
        with col:
            fig = _plot_instance(coords, demands, f"n={size}")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)


def _render_solve(solve_defaults: dict[str, Any], data_root: Path) -> None:
    st.markdown('<div class="section-label">Solve &amp; label</div>', unsafe_allow_html=True)
    _blurb(
        "Pick a generated dataset folder, configure the solver, then write labels into "
        "<code>labels/</code> inside that same folder."
    )

    runs = _discover_run_folders(data_root)
    if not runs:
        st.info("No dataset folders yet — switch to Generate first.")
        return

    labels = []
    for name in runs:
        labels.append("(legacy root)" if name == "." else name)
    choice = st.selectbox(
        "dataset folder",
        options=list(range(len(runs))),
        format_func=lambda i: labels[i],
        help="Folders under the workspace, named from generate configuration.",
    )
    run_key = runs[int(choice)]
    dataset_dir = data_root if run_key == "." else data_root / run_key
    labels_dir = dataset_dir / "labels"
    _path_chip(dataset_dir)
    st.caption(f"Labels will be written to `{labels_dir.relative_to(ROOT)}/`")

    cfg_path = dataset_dir / "config.yaml"
    if cfg_path.is_file():
        with st.expander("Dataset config", expanded=False):
            st.code(cfg_path.read_text(), language="yaml")

    available = _discover_sizes(dataset_dir)
    if not available:
        st.warning("This folder has no cvrp{N}_*.csv pairs.")
        return

    sizes = sorted(
        st.multiselect(
            "sizes to label",
            options=available,
            default=available,
            key="solve_sizes",
            help="Solved in ascending order (smallest first).",
        )
    )

    st.markdown('<div class="section-label">Solver</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        solver_name = st.selectbox(
            "engine",
            options=["pyvrp", "ortools"],
            index=0 if str(solve_defaults.get("solver", {}).get("name", "pyvrp")) == "pyvrp" else 1,
            help="PyVRP (HGS) or Google OR-Tools.",
        )
    with c2:
        seed = st.number_input(
            "seed",
            min_value=0,
            value=int(solve_defaults.get("seed", 42)),
            step=1,
            key="solve_seed",
        )

    time_limited = st.toggle(
        "limit wall-clock time",
        value=True,
        help="Off = run until no-improvement stop (PyVRP) / mapped budget (OR-Tools).",
    )
    time_limit: float | None
    no_improvement_seconds: float | None
    if time_limited:
        time_limit = float(
            st.number_input(
                "seconds / instance",
                min_value=0.1,
                value=float(solve_defaults.get("solver", {}).get("time_limit", 1.0)),
                step=0.5,
                help="Hard MaxRuntime per instance.",
            )
        )
        use_ni = st.toggle(
            "also stop on no improvement",
            value=False,
            help="Extra early stop when the best cost stalls (PyVRP wall-clock).",
        )
        no_improvement_seconds = (
            float(
                st.number_input(
                    "no-improvement seconds",
                    min_value=0.1,
                    value=5.0,
                    step=0.5,
                    key="ni_limited",
                )
            )
            if use_ni
            else None
        )
    else:
        time_limit = None
        no_improvement_seconds = float(
            st.number_input(
                "no-improvement seconds",
                min_value=0.1,
                value=10.0,
                step=0.5,
                key="ni_unlimited",
                help="Stop when the best cost has not improved for this many seconds. "
                "Required when time is unlimited (PyVRP). OR-Tools uses this as its time budget.",
            )
        )

    st.markdown('<div class="section-label">Fleet</div>', unsafe_allow_html=True)
    _blurb("Vehicle limit per dataset size. Written into every label JSON entry.")
    # Exact X needs a negative per-vehicle fixed cost; only OR-Tools supports that.
    fleet_options = (
        ["unlimited", "up_to", "exact"] if solver_name == "ortools" else ["unlimited", "up_to"]
    )
    fleet_choice = st.radio(
        "vehicles",
        options=fleet_options,
        horizontal=True,
        format_func=lambda m: {
            "unlimited": "Unlimited",
            "up_to": "Up to X",
            "exact": "Exact X",
        }[m],
        help="Unlimited ≈ one vehicle per customer. Up to X caps the fleet. "
        "Exact X (OR-Tools only) pushes the solver to use all X vehicles.",
        key=f"fleet_mode_{solver_name}",
    )
    fleet_mode: FleetMode = fleet_choice
    fleet_size_by_n: dict[int, int | None] = {size: None for size in sizes}
    if fleet_mode in ("up_to", "exact") and sizes:
        cols = st.columns(len(sizes))
        for col, size in zip(cols, sizes, strict=True):
            with col:
                # Sensible default: roughly n/4, at least 2.
                default_x = max(2, size // 4)
                fleet_size_by_n[size] = int(
                    st.number_input(
                        f"CVRP-{size}",
                        min_value=1,
                        value=default_x,
                        step=1,
                        key=f"fleet_size_{size}",
                        help=f"Fleet size X for CVRP-{size} labels.",
                    )
                )

    preview_idx = st.number_input(
        "preview instance",
        min_value=0,
        value=0,
        step=1,
        help="0-based index drawn after labeling. All instances are still solved.",
    )

    st.markdown('<div class="section-label">Labels output</div>', unsafe_allow_html=True)
    _path_chip(labels_dir)

    solve = st.button("Solve & label", type="primary", use_container_width=True)
    if not sizes:
        st.warning("Pick at least one size.")
        return
    if not solve:
        return

    labels_dir.mkdir(parents=True, exist_ok=True)
    solve_snapshot = {
        "dataset_dir": str(dataset_dir.relative_to(ROOT)),
        "labels_dir": str(labels_dir.relative_to(ROOT)),
        "sizes": list(sizes),
        "solver": solver_name,
        "seed": int(seed),
        "time_limit": time_limit,
        "no_improvement_seconds": no_improvement_seconds,
        "fleet_mode": fleet_mode,
        "fleet_size_by_n": {str(k): v for k, v in fleet_size_by_n.items()},
    }
    (labels_dir / "solve_config.yaml").write_text(yaml.safe_dump(solve_snapshot, sort_keys=False))

    progress = st.progress(0, text="Solving…")
    summary_rows: list[dict[str, Any]] = []
    preview_bundle: tuple[Any, CVRPSolution] | None = None

    for i, size in enumerate(sizes):
        progress.progress(i / max(len(sizes), 1), text=f"Starting CVRP-{size}…")
        dataset = load_dataset(dataset_dir / f"cvrp{size}")
        size_fleet = fleet_size_by_n.get(size)
        solutions = solve_dataset(
            dataset,
            solver=solver_name,
            time_limit=time_limit,
            no_improvement_seconds=no_improvement_seconds,
            fleet_mode=fleet_mode,
            fleet_size=size_fleet,
            seed=int(seed),
        )
        out = save_labels(solutions, labels_dir / f"cvrp{size}_labels.json")
        costs = [s.cost for s in solutions]
        summary_rows.append(
            {
                "size": size,
                "n": len(solutions),
                "feasible": sum(1 for s in solutions if s.feasible),
                "fleet_mode": fleet_mode,
                "fleet_size": size_fleet,
                "mean_cost": round(float(np.mean(costs)), 4),
                "vehicles": round(float(np.mean([s.num_vehicles for s in solutions])), 2),
                "runtime_s": round(float(np.mean([s.runtime_seconds for s in solutions])), 3),
                "path": out.name,
            }
        )
        # Preview the first (smallest) size so the map matches solve start order.
        if preview_bundle is None:
            idx = min(int(preview_idx), len(solutions) - 1)
            preview_bundle = (dataset[idx], solutions[idx])

    progress.progress(1.0, text="Done")
    st.success(f"Labels written under `{labels_dir.relative_to(ROOT)}`")
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    if preview_bundle is not None:
        instance, solution = preview_bundle
        dem = instance.demands
        cap = float(instance.capacity)
        cust_dem = np.delete(dem, instance.depot_index)
        st.markdown('<div class="section-label">Route preview</div>', unsafe_allow_html=True)
        _blurb(
            f"Instance <b>{solution.instance_id}</b> · cost {solution.cost:.4f} · "
            f"{solution.num_vehicles} vehicles · "
            f"{'feasible' if solution.feasible else 'infeasible'} · "
            f"capacity <b>{cap:g}</b> · "
            f"demand min/mean/max "
            f"<b>{float(cust_dem.min()):.3g}</b> / "
            f"<b>{float(cust_dem.mean()):.3g}</b> / "
            f"<b>{float(cust_dem.max()):.3g}</b> "
            f"(raw integer demands)."
        )
        fig = _plot_solution(
            instance.coords,
            instance.demands,
            cap,
            solution,
            f"{solver_name} · cap={cap:g}",
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        route_rows: list[dict[str, Any]] = []
        for r_i, route in enumerate(solution.routes):
            load = float(sum(float(dem[int(n)]) for n in route))
            route_rows.append(
                {
                    "route": r_i,
                    "customers": len(route),
                    "load": round(load, 6),
                    "capacity": cap,
                    "fill": round(load / cap, 4) if cap > 0 else None,
                    "nodes": route,
                    "demands": [round(float(dem[int(n)]), 6) for n in route],
                }
            )
        st.dataframe(route_rows, use_container_width=True, hide_index=True)


def _render_convert(data_root: Path) -> None:
    st.markdown('<div class="section-label">Convert to examples</div>', unsafe_allow_html=True)
    _blurb(
        "Turn labeled run folders into one <code>CVRPExample</code> JSON per instance "
        "(the modeling format read by <code>data.dataset.load_dataset</code>). Files are "
        "written to <code>examples/</code> inside each selected folder."
    )

    runs = _discover_run_folders(data_root)
    labeled = []
    for name in runs:
        run_dir = data_root if name == "." else data_root / name
        if _labeled_sizes(run_dir):
            labeled.append(name)
    if not labeled:
        st.info("No labeled folders yet — generate a dataset, then Solve it to create labels.")
        return

    display = ["(legacy root)" if name == "." else name for name in labeled]
    picked = st.multiselect(
        "run folders",
        options=list(range(len(labeled))),
        default=[0],
        format_func=lambda i: display[i],
        help="One or more solved folders to convert. Each keeps its own examples/ output.",
    )
    if not picked:
        st.warning("Pick at least one folder.")
        return

    for i in picked:
        run_dir = data_root if labeled[i] == "." else data_root / labeled[i]
        sizes = _labeled_sizes(run_dir)
        out_rel = (run_dir / "examples").relative_to(ROOT)
        st.caption(f"`{display[i]}` → sizes {sizes} → `{out_rel}/`")

    if not st.button("Convert to examples", type="primary", use_container_width=True):
        return

    summary_rows: list[dict[str, Any]] = []
    progress = st.progress(0, text="Converting…")
    for step, i in enumerate(picked):
        name = labeled[i]
        run_dir = data_root if name == "." else data_root / name
        progress.progress(step / max(len(picked), 1), text=f"Converting {display[i]}…")
        try:
            written = export_run(run_dir)
        except Exception as exc:  # surface any per-folder failure in the UI
            summary_rows.append(
                {"folder": display[i], "size": "—", "examples": 0, "status": f"error: {exc}"}
            )
            continue
        out_dir = run_dir / "examples"
        for size, paths in written.items():
            summary_rows.append(
                {
                    "folder": display[i],
                    "size": size,
                    "examples": len(paths),
                    "status": f"ok → {out_dir.relative_to(ROOT)}/",
                }
            )

    progress.progress(1.0, text="Done")
    total = sum(int(row["examples"]) for row in summary_rows)
    st.success(f"Wrote {total} example JSON file(s).")
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)


def main() -> None:
    gen_defaults = _load_yaml(DEFAULT_GEN_CFG)
    solve_defaults = _load_yaml(DEFAULT_SOLVE_CFG)

    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">CVRP <span>Dataset Generator</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "mode",
        options=["Generate", "Solve", "Convert"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    _blurb("Root folder for generated runs (each config gets its own subfolder).")
    data_root = _resolve(
        st.text_input(
            "workspace",
            value=str(DEFAULT_DATA.relative_to(ROOT)),
            label_visibility="collapsed",
            placeholder="data/raw/cvrp",
        )
    )
    _path_chip(data_root)

    if mode == "Generate":
        _render_generate(gen_defaults, data_root)
    elif mode == "Solve":
        _render_solve(solve_defaults, data_root)
    else:
        _render_convert(data_root)


if __name__ == "__main__":
    main()
