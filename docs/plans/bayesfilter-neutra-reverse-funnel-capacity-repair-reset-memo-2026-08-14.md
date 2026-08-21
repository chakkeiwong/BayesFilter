# NeuTra reverse-funnel capacity repair reset memo (2026-08-14)

## Current state

The capacity-repair campaign is complete. No process is running. GPU 1 returned
to idle after the final run; GPU 0 was not used.

The original scale-cap hypothesis was only partly correct. Hard-capped stages
cannot exactly represent `log scale = y`. The new optional
`stage_unbounded_scale_linear` route repairs that mathematical defect and is
identity-initialized, strictly autoregressive, state-serializable, and covered
by exact-map and derivative tests. It remains disabled by default.

The matched corrected arm did not pass proposal law, so no HMC was launched.
Do not describe the trained state as a valid NeuTra transport or continue to HMC
from it.

The August 15 root-cause trace supersedes the earlier coefficient-only
interpretation. Full reversal moves `y` to the final autoregressive coordinate
in stage 1, allowing that stage to condition the root map on all `x`
coordinates. Stage 1 creates most of the observed root variance and tail loss.
The stages then co-adapt, so forcing only the additive first row to one is not a
valid repair of the frozen composition. See
`docs/plans/bayesfilter-neutra-reverse-funnel-root-cause-diagnosis-2026-08-15.md`.

## Important paths

- Plan:
  `docs/plans/bayesfilter-neutra-reverse-funnel-capacity-repair-plan-2026-08-14.md`
- Result:
  `docs/plans/bayesfilter-neutra-reverse-funnel-capacity-repair-result-2026-08-14.md`
- Core implementation:
  `bayesfilter/inference/neutra_weighted_training.py`
- Runner:
  `docs/benchmarks/run_neutra_reverse_funnel_capacity_2026_08_14.py`
- Corrected artifacts:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/reverse-funnel-capacity-r3/`
- Rejected pre-cap diagnostic:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/reverse-funnel-capacity-r2/`

## Scientific handoff

The next smallest discriminating research question is whether a root-preserving
ordering prevents the reverse-KL tail-collapse shortcut. A new plan should
compare:

1. the exact one-stage additive scale matrix with the root fixed to identity;
2. residual stages with permutations restricted to coordinates `1:100` and the
   root map still protected; and
3. joint fine-tuning from that warm start under a decaying LR/convergence rule.

Use the same exact proposal-law authority. In the one-stage constrained arm the
first-row coefficients should learn toward one; coefficient error can be
explanatory or an early repair trigger, but proposal law must remain the HMC
nomination gate. Do not increase generic network width/depth or simply extend
the full-reversal joint run.

## Engineering handoff

Broadened focused tests currently pass: `43 passed`. The implementation preserves old
artifacts because both new configuration tuples default to empty/disabled.
Pre-cap and additive unbounded modes are distinct and mutually exclusive per
stage, preventing archived states from being reinterpreted under new semantics.

The repository is heavily dirty due to concurrent work. Only the files and
artifact roots named in the result note belong to this lane; preserve all other
changes.
