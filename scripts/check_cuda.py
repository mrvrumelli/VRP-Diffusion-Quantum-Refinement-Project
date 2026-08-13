"""Report CUDA readiness and optionally require a successful GPU backward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrp_diffusion_quantum.utils.cuda import cuda_diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit non-zero unless CUDA is available and the backward smoke check passes",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = cuda_diagnostics(run_backward=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.require_cuda and not (report["cuda_available"] and report["backward_smoke_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
