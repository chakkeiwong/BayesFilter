# LGSSM NeuTra Gap Closure Phase 4 Result - HMC Tuning Admission

Date: 2026-07-15  
Decision: `SUPERSEDED_FIXED_BUDGET_ADMISSION_DESIGN`

## 2026-07-15 Correction

This result is preserved as historical execution evidence but its terminal
candidate rejection is invalid. The Phase 4 harness fixed burn-in and retained
sampling at 1,000 transitions per chain, discarded burn-in, and rejected a
candidate at the first modern R-hat miss. The corrected contract retains and
diagnoses sequential warm-up, then extends retained sampling until the modern
R-hat gate passes or 10,000 draws per chain are collected. See
`docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-repair-plan-2026-07-15.md`.

The numerical values below remain valid descriptions of the old fixed run.
They do not establish rejection under the corrected adaptive-budget policy.

## Outcome

Both fresh 5,000-step frozen candidates completed the complete predeclared HMC
tuning procedure: primary five-point grid, one seven-point repair grid, and a
fresh 1,000-result-per-chain verification for the repair nominee. Both tuning
routes were TensorFlow/TFP, CPU-hidden, float64, XLA, and batched across four
chains. Both candidates were finite, moved all chains, had valid target-status
telemetry, and had zero energy-error divergence screens. Neither passed the
required maximum of rank-normalized and folded rank-normalized split R-hat
`<=1.01`.

| Candidate | Repair step | Verification acceptance | Max rank R-hat | Max folded R-hat | Failing parameter | Health/status/divergence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `dense_seed1201` | `0.8` | `0.69275` | `1.00547` | `1.01569` | `a22_raw` | pass / 0 invalid / 0 |
| `dense_seed1202` | `0.8` | `0.70425` | `1.01721` | `1.00638` | `a42_raw` | pass / 0 invalid / 0 |

Primary result artifact:
`docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/phase4/result.json`
with SHA-256
`5c1b32254e247444313afcdb99170008b426a4ed18383c8ac0a4ff6e7dc6bf9c`.

Candidate artifacts:

- `phase4/dense_seed1201/result.json`, artifact hash
  `sha256:ec870c42abe1bd0d477947fe36ef2e2b6faacffbf88aa338b25842f37c2029d5`;
- `phase4/dense_seed1202/result.json`, artifact hash
  `sha256:4c9524e173cfaa39cebcca6f59ac38af45ff83196b42709e900542db74926299`.

## Interpretation

Under the old design this was classified as candidate/tuning/diagnostic
failure, not a target, training, artifact, or research-direction failure:

- Phase 1/2 training engineering passed for both seeds;
- Phase 3 frozen GPU/CPU objective parity passed for both seeds;
- HMC health, finite-state, target-status, and energy-error screens passed;
- only the declared modern R-hat admission criterion failed.

Acceptance was near the intended `0.70` in both verification runs. That does
not rescue either candidate: acceptance is a nomination diagnostic, while
rank/folded R-hat is the admission criterion.

## Decision Tables

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not admit from the fixed 1,000-draw artifact alone | max modern R-hat must be `<=1.01` for all 18 parameters | seed1201 folded R-hat `1.01569`; seed1202 rank R-hat `1.01721` | whether sequential extension passes before 10,000 draws | run the corrected fresh sequential controller | NeuTra direction is not disproved; no posterior correctness, superiority, calibration, robustness, or default claim |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | no health/identity hard veto; fixed draw budget was insufficient for admission |
| Statistically supported ranking | none |
| Descriptive-only differences | seed-specific failing parameter and R-hat magnitudes; no seed ranking |
| Default readiness | not evaluated |
| Next evidence needed | a new explicitly reviewed repair contract that addresses R-hat failure without weakening the gate |

## Budget And Stop Record

Tuning consumed approximately `1270.88 s` of CPU wall time across both
candidates, within the six-hour tuning budget. The old plan's declared grid
repair was exhausted. No confirmatory HMC, posterior-agreement, or recovery
claim was executed. The claimed continuation veto is superseded because it
arose from the fixed-budget planning error, not a target, implementation,
health, or 10,000-draw cap failure.
