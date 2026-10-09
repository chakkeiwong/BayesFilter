# NeuTra gap closure: reset memo

The requested code/math audit, reviewed repair plan and bounded execution are
complete. All three previously unresolved simpler-model cases now have fresh
passing sequential HMC and final-reference results. No worker remains active.

- Plan: `bayesfilter-neutra-gap-closure-plan-2026-10-01.md`.
- Findings, derivations, failed arms and final numerical tables:
  `bayesfilter-neutra-gap-closure-results-2026-10-01.md`.
- Machine-readable terminal review:
  `artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/gap-closure-20261001-r1/terminal-review.json`.
- Controller: `scripts/run_neutra_gap_closure.py`; the continuation-aware
  resume entry point is `scripts/continue_neutra_gap_closure.py`. A completed
  resume was exercised and launched no new workers.

## Confirmed scope and selected results

The mixture's frozen sampler passed from saved walkers and both modes; actual
adaptive training rows had the correct observed mode weights and valley mass.
Clipping was zero. Checked forward-loss directional derivatives agreed with
finite differences at about 1e-11. The weak .02 initializer nevertheless
remained near the derived Gaussian stationary configuration through 65,536
updates. Variance .2 produced useful nonlinear fits; seed11 also required the
tested width16 configuration to meet the warm-fit screen within this ladder.
These findings do not establish a local-minimum theorem or a universal setting.

| Case | Selected map | Selected HMC | Retained/chain | Qualification directory under campaign-r1/attempts |
|---|---|---|---:|---|
| Mixture/Gabrié seed11 | Width16, variance .2; 65,536 FKL + 2,048 RKL | ε=.25, L=9 | 2,000 | `gap-20261001-init-w16-v0.2-s11-qualify-v0-r1` |
| Mixture/Gabrié seed37 | Width8, variance .2; 65,536 FKL, warm map selected | ε=.25, L=9 | 2,000 | `gap-20261001-init-w8-v0.2-s37-qualify-v0-r1` |
| Wiggle/Gabrié seed23 | Preserved map | ε=.1767766953, L=9 | 6,000 | `gap-20261001-wiggle-gabrie-s23-qualify-v0-r1` |

Each selected member used 2,000 warmup transitions per chain and passed the
unchanged numerical, R-hat, ESS, MCSE/SD and final-reference screens. Each first
member failed before the second passed; all member archives remain preserved.
The final reference was opened once after selection, not used to select among
members. Identity mass remained fixed in latent coordinates.

All 30 original target/arm/seed groups now have a preserved passing execution.
Only the three cases above were freshly qualified in this cycle. Other groups
retain earlier evidence and policies; this collection does not estimate a
success probability or establish a statistically supported method ranking.

## Engineering and resource state

Implemented actual-row training diagnostics, optional initializer variance,
bounded verified-member selection, final-reference separation and a conditional
continuation for supported nonlinear progress. The latter was unnecessary once
both mixture seeds qualified; its scheduler and state-restoration boundaries
were tested. Future workers also use an attempt-local launch config. Exact
configs for these already completed workers were recovered and verified against
their original pre-run hashes without altering their original manifests.

54 distinct focused regression tests passed across the recorded suites. The
terminal artifact review checked 628 saved source files, 155 inputs, 22 exact
configs and 12 new 1,000-point probes. The execution used the preserved Python
source snapshot, the existing tfgpu environment, XLA, GPU1 and verified memory
growth. GPUs0/2 were left to their existing jobs.

There were 19 GPU workers and three CPU reference workers, no infrastructure
retry, and no new budget allocation. Cost: 1.062799 GPU process-hours and
1.266186 CPU core-hours. Remaining shared conservative allocation: 22.969666
GPU process-hours and 47.867560 CPU core-hours. The 6/12-hour repair sub-cap
was respected. Routine test time is recorded separately in the result note.

Do not resume the old Gaussian controls or spend the remaining budget merely
because it exists. No numerical work remains in this reviewed cycle. Further
q20 or new-target work requires its own target-specific training and downstream
evidence plan. The selected maps still have large score-residual tails, so do
not describe them as uniformly whitened, production-certified, or proof of
universal NeuTra reliability.

The shared worktree contains substantial unrelated changes. No commit, merge,
push, package installation or external publication was performed in this repair.
