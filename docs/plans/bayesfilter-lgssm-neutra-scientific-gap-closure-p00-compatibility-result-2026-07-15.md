# LGSSM NeuTra Gap Closure Phase 0 Result

Date: 2026-07-15  
Decision: `PASS_PHASE0_COMPATIBILITY_GATE`

## Outcome

The stale execution join was repaired before long GPU training. The new active
campaign route is TensorFlow/TFP-only, consumes the current strict training
schema, binds the immutable selected recipe and tuned plain-HMC summary, runs
four chains in one XLA-compiled `tfp.mcmc.sample_chain` call, uses modern
rank/folded split R-hat, serializes tensors through TensorFlow, and does not
import the legacy NumPy-backed HMC/tuning/orchestration modules.

## Checks And Evidence

| Check | Result |
| --- | --- |
| Focused unit/fixture tests | `9 passed, 2 warnings` |
| Static selected recipe | pass; `wide_2x_lr5e3`, file SHA-256 `1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97` |
| Plain-HMC comparator summary | pass; file SHA-256 `bcc6e71a1067dc648758a5aac9c87ef7e94fdd4b1ac53d5601ef4e9fdf6741b5` |
| New-route import closure | pass; no NumPy, legacy HMC, legacy tuner, old orchestrator, or TensorFlow host callback import |
| Python compile | pass |
| `git diff --check` | pass |
| Gaussian CPU/XLA HMC | pass; 4 chains, 128 results/chain, all chains moved, zero energy-error divergences, max modern R-hat `1.003883` |
| Historical wide-screen frozen HMC integration | pass; exact 18D target, 4 batched chains, CPU/XLA, all finite/status valid, all chains moved |
| Tensor archive | exact TensorFlow serialize/parse round trip and no-overwrite test pass |
| Admission semantics | acceptance-only nomination tested; fewer than 1,000 verification draws and folded R-hat scale mismatch both block admission |

Runtime evidence:

- `docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/phase0/gaussian_xla_hmc.json`
- `docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/phase0/historical_screen_frozen_hmc_smoke.json`

The CPU-hidden runs logged CUDA initialization failure after GPU hiding. This
is expected for deliberate `CUDA_VISIBLE_DEVICES=-1` execution and is not GPU
stack evidence. Both runs compiled and executed on the XLA Host service.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 0 to seed1201 training | policy-compliant execution join passes focused mechanics, schema, diagnostic, and artifact gates | no identity, finite, XLA, movement, status, serialization, or import-closure veto | historical frozen smoke is only a 500-step structural fixture | run fresh 5,000-step seed1201 on trusted GPU/XLA | no LGSSM NeuTra posterior or training-quality claim |

## Repair Record

The static audit first used the old retained-archive SHA-256 where the new route
needed the final comparator-summary SHA-256. It also initially treated
`selected_recipe_source` as a direct file reference rather than its actual
nested strict-schema form. Both were corrected before any long run, and the
focused suite was rerun successfully.

## Handoff

Phase 1 may start under
`docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-p01-seed1201-subplan-2026-07-15.md`.
The new module's frozen-candidate path remains to be exercised against the
fresh 5,000-step payload before Phase 3 can close.
