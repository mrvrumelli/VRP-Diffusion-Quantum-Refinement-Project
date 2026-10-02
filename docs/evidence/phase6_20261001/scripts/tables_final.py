"""Markdown tables for the final test section, from final/analysis_{reserved,ood}.json."""

import json
from pathlib import Path

FINAL = Path(__file__).resolve().parent / "final"
LABELS = [
    ("T1", "Classical, one pass, 5/8/10"),
    ("T2", "Classical, 10 restarts, 5/8/30"),
    ("T3", "Classical, 10 restarts, 10/16/30"),
    ("T4", "SA on QUBO, 5/8/30"),
    ("T5", "SA, then classical 10 restarts, 5/8/30"),
    ("T8", "Classical, 10 restarts, 4/6/10"),
    ("T10", "Classical exact, 4/6/10"),
    ("T6", "QAOA depth 1, 4/6/10"),
    ("T7", "Random QUBO sampling, 4/6/10"),
    ("T9", "QAOA, then classical 10 restarts, 4/6/10"),
]
SETS = [("reserved", "Reserved test"), ("ood", "R/C/RC cells")]


def fmt(value):
    mean, (low, high) = value
    return f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"


def main() -> None:
    data = {s: json.loads((FINAL / f"analysis_{s}.json").read_text()) for s, _ in SETS if (FINAL / f"analysis_{s}.json").exists()}
    print("| Configuration | " + " | ".join(f"{title}: N20 / N50 / N100 / all" for s, title in SETS if s in data) + " | Solver s per graph, reserved |")
    print("|---|" + "---:|" * (len(data) + 1))
    base = " | ".join(
        " / ".join(f"{data[s]['configs'][n]['baseline']:.2f}" for n in ("20", "50", "100", "all")) for s, _ in SETS if s in data
    )
    print(f"| **Before refinement** | {base} | — |")
    for key, label in LABELS:
        cells = []
        for s, _ in SETS:
            if s not in data:
                continue
            c = data[s]["configs"]
            cells.append(" / ".join(f"{c[n][key]['gap']:.2f}" if key in c[n] else "—" for n in ("20", "50", "100", "all")))
        seconds = data["reserved"]["configs"]["all"].get(key, {}).get("solver_seconds", float("nan"))
        print(f"| {label} | " + " | ".join(cells) + f" | {seconds:.0f} |")
    print()
    print("| Pre-declared comparison, all graphs | " + " | ".join(title for s, title in SETS if s in data) + " |")
    print("|---|" + "---:|" * len(data))
    labels = list(next(iter(data.values()))["comparisons"]["all"].keys())
    for label in labels:
        cells = [fmt(data[s]["comparisons"]["all"][label]) if label in data[s]["comparisons"]["all"] else "—" for s, _ in SETS if s in data]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
