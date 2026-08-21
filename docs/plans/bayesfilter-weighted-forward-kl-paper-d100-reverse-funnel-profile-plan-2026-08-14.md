# Reverse-funnel NeuTra training profile plan (2026-08-14)

Status: `COMPLETE`

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | What GPU/XLA time and operator work does the frozen reverse-KL d100 funnel training path spend per update and heldout objective evaluation? |
| Candidate under test | Existing frozen `paper_funnel`, `reverse_kl`, 100-dimensional `(100,100)` three-stage ELU transport at update 5000. |
| Comparator | Same TensorFlow/XLA trainer and frozen-state loading path; no objective or architecture change. |
| Primary result | Wall time for one compiled reverse-KL update and one 65,536-row heldout reverse-KL evaluation, plus TensorFlow profiler trace. |
| Hard vetoes | GPU memory-growth failure, wrong target/objective/state hash, nonfinite values, missing XSpace, or a non-XLA execution. |
| Explanatory diagnostics | Gradient norm, clipping status, target/logdet finiteness, GPU memory policy, and profiler operator trace. |
| Nonclaims | Runtime does not establish reverse-KL convergence, posterior correctness, objective ranking, or a repair. |

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early check |
|---|---|---|---|
| GPU 1 | Existing d100 training target and current device policy | Device contention or unhealthy GPU | trusted `nvidia-smi`, visible logical-device check |
| `TF_FORCE_GPU_ALLOW_GROWTH=true` and float64/TF32 off | repository GPU policy and frozen d100 manifest | launch-invalid allocator or unintended TF32 | memory-policy receipt and runtime manifest |
| Batch 4096 | frozen reverse training manifest | profile does not represent claim run | exact config/state check |
| Heldout size 65,536 | frozen replay selection count and d100 profile comparator | evaluation cost not comparable | static shape check |
| One warmup update before profiling | diagnostic compilation warmup | trace includes one-time compilation | run one unprofiled update and one unprofiled evaluation first |

## Skeptical audit

- The existing profiler cannot answer this question because it hard-codes
  forward-KL Gaussian training. A separate reverse-funnel runner is required.
- A successful profiler only establishes runtime behavior; it cannot explain
  the reverse-funnel tail failure without the already completed proposal and
  HMC diagnostics.
- The profile must load the frozen state and assert objective/target identity,
  otherwise it could silently profile a different arm.
- The gateway smoke profile already passed on GPU 1; this reverse profile is a
  fresh diagnostic root and does not overwrite any scientific artifact.

Audit verdict: `PASS_FOR_EXECUTION`.

## Planned artifact

`docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/reverse-funnel-profile-r1/`

## Execution record

The gateway retry and reverse-funnel profile were executed on GPU 1 with
TensorFlow memory growth, float64, TF32 disabled, and XLA enabled.

1. The existing Gaussian forward-KL profiler completed in fresh root
   `paper-d100/forward-profile-r2` with no `502`; it produced a CUPTI/XSpace
   trace. This was a gateway smoke test only because that legacy profiler
   mutates the frozen weights during its warmup update.
2. The first reverse-funnel profile completed in `reverse-funnel-profile-r1`
   with no `502`, but its warmup update mutated the frozen state and used the
   stored base learning rate. It is retained as infrastructure/debug evidence
   and is not used for timing interpretation.
3. A repaired profile completed in
   `paper-d100/reverse-funnel-profile-r3`. It restores model and optimizer
   variables after compilation warmup, uses the final schedule learning rate
   `1e-4`, and profiles the deterministic update-5000 batch. Its artifact
   ledger verifies the result and XSpace file.

Valid `r3` measurements:

| Quantity | Result |
|---|---:|
| Reverse-KL update wall time | `0.01243 s` |
| 65,536-row heldout reverse-KL evaluation | `0.04897 s` |
| Heldout reverse-KL before profiled update | `50.09316` |
| Gradient norm | `1.04834` |
| Clipping on profiled update | `false` |
| XLA/CUPTI trace | present, `581642` bytes |

The optimizer state in the frozen training artifact did not include Adam
moments. The update timing is therefore a zero-slot optimizer timing
diagnostic; the heldout evaluation timing and target/operator profile are not
affected by that omission.

The gateway is no longer returning the prior `502` error. The reverse-funnel
profile is runtime evidence only; it does not establish that reverse KL has
converged or that its posterior tails are correct.
