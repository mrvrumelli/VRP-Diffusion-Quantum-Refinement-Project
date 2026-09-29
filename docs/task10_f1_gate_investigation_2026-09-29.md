# Track B F1 gate investigation — 2026-09-29 (paused)

Full account of diagnosing why `paper_cmd` diffusion training can't clear the paper's ≈0.823
sample-F1 target, after the full-scale training run
([`task10_paper_cmd_diffusion_full_2026-09-28.md`](task10_paper_cmd_diffusion_full_2026-09-28.md))
came back *worse* than the pilot despite 11x more data — directly contradicting the pilot's own
"just needs more data" diagnosis.

## Best result overall

**0.505** (the original pilot, 2,886 examples) — no later, larger, or differently-composed
training run has beaten this. Target: **≈0.823**.

## Hypotheses tested, in order, all ruled out or structurally blocked

| # | Hypothesis | Test | Confirmed result | Verdict |
|---|---|---|---|---|
| 1 | The pilot-vs-full gap is measurement noise (training loop only samples 8/size) | Re-scored the full-scale checkpoint on a large, fixed 1,500-example panel | **0.4707**, tight 95% CI | Ruled out — the gap is real |
| 2 | Stability-filtering biases the training distribution toward "easy" instances | Retrained on the fully unfiltered label pool (45,522 examples, zero stability filtering) | **0.4731** on the same large panel — statistically indistinguishable from filtered | Ruled out |
| 3 | 50 inference steps isn't enough to reach a good reconstruction | Re-scored the same checkpoint at 10/20/50/200/1000 steps | 0.493 → 0.479 → 0.471 → 0.464 → 0.463 — **flat to slightly declining** | Ruled out — more steps doesn't help, if anything hurts slightly |
| 4 | N50/N100's noisier labels drag the aggregate F1 down; N20 alone would do much better | Trained and confirmed a dedicated N20-only model (15,080 clean examples) on a large 1,341-example panel | **0.5389** — essentially the same as N20's share of the mixed-size result (0.547) | Ruled out — isolating the cleanest size in isolation doesn't meaningfully help |
| 5 | A specific training hyperparameter (augmentation, LR, batch size, epochs) is mistuned | Attempted adding standard geometric augmentation on top of the paper recipe | **Structurally blocked**: `paper_cmd`'s alignment contract hard-locks every training hyperparameter to the paper's stated values (`ValueError: paper_cmd diffusion contract mismatch: training.augmentation=True (expected False)`) | Can't be tested without leaving the `paper_cmd` track entirely |

## What this rules in

Five real, independent explanations have now been eliminated or found untestable within the
track's own rules. What's left, unchanged from the original report's candidate list but now with
much more confidence behind it:

1. **An unresolved paper ambiguity is the most likely remaining candidate.**
   [`cmd_paper_comparison_contract.md`](cmd_paper_comparison_contract.md) already lists the exact
   HGS labeling budget, seed count, and augmentation enumeration as open author questions. This
   project's 2-seed protocol and assumed 10/20/40s time budgets (both documented assumptions, not
   confirmed facts) may simply produce systematically different-quality training targets than
   whatever the paper's authors actually used — and that's not fixable by anything on this
   project's side without either author clarification or a real guess-and-check on solve time
   budgets specifically (a new, separate labeling investment, not yet attempted).
2. **A structural property of this exact architecture+labeling combination** that the paper_cmd
   contract's strictness makes untestable from inside the track (e.g., is the frozen GAT genuinely
   sufficient, or would this exact data reach much higher F1 with a trainable encoder — the paper's
   own authors reported this tradeoff exists, just not this large). Testing this would mean an
   explicitly-labeled `ours_robust`-track comparison, a real scope decision, not a quick check.

## Not investigated further, by explicit instruction

Per direction received during this investigation: **paused here rather than continuing into either
of the two remaining candidates** (both are real scope/resource decisions — new labeling budget
experiments, or a track-crossing architecture comparison — not more of the same quick diagnostic
loop). Autonomous work is redirected to Track A's untouched backlog (Task 5) until further
direction.

## Everything that stayed correct throughout

All 8 training runs across this investigation (pilot, full-scale filtered, full-scale unfiltered,
N20-only baseline, N20-only+augmentation attempt) that actually ran completed with 100% capacity
feasibility where routes were decoded, zero solver errors, and clean quality gates (`ruff`/`mypy`/
`pytest`) throughout. This is a genuine research finding, not an engineering failure — the paper's
own contract-enforcement machinery worked exactly as designed by refusing to silently let a
hyperparameter drift.
