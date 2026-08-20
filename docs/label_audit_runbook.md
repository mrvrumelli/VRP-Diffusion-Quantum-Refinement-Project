# Running the s7799 strong-label audit

Step-by-step instructions for a new contributor to run
`scripts/run_strong_label_audit.py`. Read `docs/label_audit_s7799_decision.md` first for what
the audit already found and decided; this document only covers executing it.

## What is already done

The audit's 1,500-instance input pool (500 each of CVRP20/50/100, deterministically selected
with seed `4201` from the 66,667-per-size source corpus) is committed at
`data/processed/label_audit_s7799/`. Regenerating it needs the 18 GB
`cvrp_s7799_n20-50-100_x66667/` corpus, which is not in git — so this commit lets you skip
copying that corpus just to run the audit. `data/processed/label_audit_s7799/subset_manifest.json`
records the exact source hash and seed if you want to verify it against your own copy of the
corpus.

The actual audit — 6,000 PyVRP runs plus 446
OR-Tools challenger runs, ~140,500 + 14,600 CPU-seconds — is the CPU-heavy part. Its completed
cache (`outputs/label_audit/s7799_strong_reference/`) is force-committed despite `outputs/` being
gitignored, so you do not need to re-run the solves. `outputs/` stays gitignored for everything
else; only this one completed run was explicitly checked in to save other contributors the CPU
time.

If you just want the final artifacts, you are done — skip to
[What comes out](#5-what-comes-out) below. Only continue with steps 1-4 if you want to reproduce
the run yourself (e.g. to sanity-check it, or because the policy config changed and you need a
fresh pass). If you do, the same resume behavior in step 4 applies: `run_strong_label_audit.py`
will see the committed candidates already in `outputs/label_audit/s7799_strong_reference/candidates/`
and skip them, only computing what's missing.

## 1. Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

This installs `pyvrp` and `ortools`, both of which the audit needs. You do **not** need the
`cvrp_s7799_n20-50-100_x66667/` corpus for this audit specifically — only for other dataset
commands in the main README.

Optional: confirm the committed input pool matches what you expect before spending CPU time on it.

```bash
python -c "
from vrp_diffusion_quantum.utils.experiment import hash_dataset
print(hash_dataset('data/processed/label_audit_s7799'))
"
```

The `input_sha256` your run records in `outputs/.../config.json` should match this. It will not
match the `Input SHA-256` line in `docs/label_audit_s7799_decision.md` — that hash was computed
on the machine that ran the original audit, and `subset_manifest.json` embeds that machine's
absolute source path, so the directory-level hash differs even though the 1,500 example files
are byte-identical.

## 2. Run the audit

```bash
python scripts/run_strong_label_audit.py
```

This uses `configs/data/label_audit_strong_s7799.yaml`: four deterministic PyVRP seeds per
instance at 10/20/40-second budgets for CVRP20/50/100, then an OR-Tools Guided Local Search
challenger on every cost-unstable or matrix-ambiguous case plus 50 deterministic stable controls
per size.

Override worker count without touching the solver policy:

```bash
python scripts/run_strong_label_audit.py --workers 8
```

By default it uses `min(cpu_count - 1, 16)`.

## 3. Expect a long, CPU-bound run

The committed policy is ~140,000 CPU-seconds of PyVRP solving alone (500 × 4 seeds ×
(10s + 20s + 40s)), plus the OR-Tools challenger pass. On an 8-core machine at the default
worker count, budget several hours of wall time. This does not benefit from a GPU — don't run it
on a machine you need for CUDA training, and don't run it alongside another CPU-heavy labeling
job or the numbers will be distorted by contention.

## 4. Interruptions are safe — the same command resumes

Every solver candidate is cached atomically under
`outputs/label_audit/s7799_strong_reference/candidates/` as soon as it finishes. If the run is
killed, closes with your laptop, or you just want to stop for the day, re-running the exact same
command skips every already-cached candidate and only computes what's missing. Do not point a
resumed run at a different `--config` or a hand-edited output directory — the run refuses to
resume into an output directory whose recorded config no longer matches.

## 5. What comes out

Under `outputs/label_audit/s7799_strong_reference/`:

- `config.json` / `config.yaml` — the exact policy and input hash for this run.
- `instance_results/` — one JSON per source instance with the full analysis.
- `reference_examples/` — the accepted best-cost reference for every instance that has one.
- `accepted_matrix_examples/` — the subset of those references whose binary route-membership
  matrix also passed the stability check (this is the stricter, trainable-label set).
- `summary.csv`, `metrics.json`, `run.log` — aggregate numbers and a plain-text run summary.

## 6. After the audit finishes

Compare your `metrics.json` against the numbers in `docs/label_audit_s7799_decision.md` (audit
integrity and per-size table). They should match closely — small differences are expected only if
the source corpus or PyVRP/OR-Tools versions differ from the original run.

The decision doc also describes the bounded CVRP100 80-/120-second follow-ups and the policy-v2
label materialization that build on this audit. Do not start those, and do not start full model
training, until you've read that document's acceptance gates and Next-stage research plan in
`docs/3060ti_training_todo.md`.

## 7. Commit your results so nobody re-solves them

Whatever you run from this doc — this audit, the CVRP100 follow-ups, the rc_pilot/rc_full pools,
or a from-scratch rerun — force-commit its output directory the same way
`s7799_strong_reference` was committed, since `outputs/` is gitignored by default:

```bash
git add -f outputs/label_audit/<your_run_name>/
git commit -m "..."
git push
```

Do this even if multiple people are working the same config in parallel: whoever finishes first
should push, and everyone else should `git pull` before continuing rather than keep solving —
the resume behavior in step 4 means their next run will pick up the pushed candidates and only
compute what's genuinely still missing. Commit a partial `candidates/` cache too if you have to
stop mid-run; a partial cache still saves the next person real CPU time, and you can commit again
once it finishes.
