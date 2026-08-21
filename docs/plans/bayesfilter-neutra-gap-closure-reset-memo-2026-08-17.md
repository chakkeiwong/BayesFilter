# NeuTra Gap-Closure Reset Memo (2026-08-17)

## State

The three-mode value/score and transport mechanics are locally correct. The
frozen transport diagnosed here is the obsolete 1,000-update small baseline,
and it does not mix between modes under the tested fixed kernels. The tuning
veto applies to that checkpoint only. The previously completed six-stage
capacity repair already passed downstream HMC and exact component-law screens.

## Frozen Evidence

- Plan: `docs/plans/bayesfilter-neutra-gap-closure-plan-2026-08-17.md`
- Result: `docs/plans/bayesfilter-neutra-gap-closure-result-2026-08-17.md`
- H2/H4: `docs/plans/artifacts/neutra-gap-closure-2026-08-17-r1/h2-h4-gpu/`
- H1/H3: `docs/plans/artifacts/neutra-gap-closure-2026-08-17-r1/h1-h3-gpu/`
- Failed three-mode tuning: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/three-mode-full/tuning/tuning_result.json`

## Constraints

- Do not relax the modern R-hat threshold or reinterpret acceptance as mode
  mixing.
- Do not use short local chains as posterior evidence.
- Do not launch HMC from the failed small checkpoint.
- Fresh six-stage replicas must pass their own support, tuning, sequential, and
  exact-law gates before contributing replication evidence.
- Keep all future training target-specific, GPU/XLA, batch-native, and
  separately hashed.

## Next Entry Point

Use
`bayesfilter-neutra-three-mode-provenance-and-evidence-closure-result-2026-08-17.md`
as the current authority. Two fresh component-aware replicas passed the full
downstream path. The zero-centered Student-t mode-blind proposal failed support
and stopped before training/HMC. The next proposal must discover separated
regions using target queries without exact component centers. Geometry and
application targets remain separate target-specific evidence lanes.
