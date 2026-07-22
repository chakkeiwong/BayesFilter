# Phase 3 Serious Campaign Launch Record

Campaign: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Date: 2026-07-18

Status: `READY_TO_LAUNCH`

## Pre-launch implementation review

The plan, registry, orchestration module, CLI, tests, Phase 1 review, and Phase
2 preflight were rechecked before launch.

- Current direct target signatures and reviewed batch-native callables pass for
  `LGSSM-EXACT`, `PP-UKF`, `PP-SGQF`, `SIR-SGQF`, and `STR-UKF`.
- Training, held-out evaluation, tuning, and HMC share the inspected batch
  target surface. Historical typed identities are provenance only.
- Common held-out stateless batches are disjoint from training seeds.
- Screen weights are never reused by final training.
- The new code imports no NumPy or historical benchmark module and implements
  no sampler, HMC kernel, R-hat, ESS, or manual kernel selector.
- Native tuning is fixed at target acceptance 0.70, band `[0.65,0.75]`, fixed
  identity mass in `z`, no fixed-grid repair, and fresh modern-R-hat
  verification.
- Warm-up and retained sampling use physical coordinates, separate archives,
  modern rank/folded R-hat, ESS, health/status/energy vetoes, and 10,000 caps.
- Truth-tail evaluation is suppressed when sampler validity fails.
- The CLI launches only executable cells, in separate processes, and includes
  all seven blocked cells only in the aggregate inventory.

Verdict: `PASS_TO_LAUNCH`.

## Launch command

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl \
python docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action campaign \
  --output-root docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-01
```

The command is authorized by the user's request to execute this plan under the
repository serious-campaign policy. It must be supervised to terminal aggregate
result or a structured continuation veto.
