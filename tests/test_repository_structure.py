from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRECTORIES = [
    "configs/data",
    "configs/diffusion",
    "configs/eval",
    "configs/policy",
    "configs/quantum",
    "configs/train",
    "data/processed",
    "data/raw",
    "data/samples",
    "docs",
    "docs/reports",
    "notebooks/sanity_checks",
    "scripts",
    "src/vrp_diffusion_quantum/data",
    "src/vrp_diffusion_quantum/diffusion",
    "src/vrp_diffusion_quantum/eval",
    "src/vrp_diffusion_quantum/inference",
    "src/vrp_diffusion_quantum/local_search",
    "src/vrp_diffusion_quantum/models",
    "src/vrp_diffusion_quantum/quantum",
    "src/vrp_diffusion_quantum/train",
    "src/vrp_diffusion_quantum/utils",
    "tests",
]


def test_repository_structure_exists() -> None:
    missing = [path for path in EXPECTED_DIRECTORIES if not (ROOT / path).is_dir()]

    assert missing == []
