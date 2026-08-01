# P4 Subplan: Predator-Prey UKF And SGQF

Phase objective: test corrected neural-force HMC independently on the admitted
predator-prey UKF and SGQF filter posteriors.

Entry conditions: P3 justifies continuation; both target/chart identities and
prior/data/filter settings replay; 24-GPU-hour combined ceiling is frozen.

Refreshed entry evidence: P3 confirms corrected learned-force validity on one
exact-likelihood LGSSM fixture, but the zero-residual chart arm was
descriptively more efficient than the learned arm. P4 therefore treats
residual learning as a target-specific hypothesis and keeps zero residual as a
co-primary validity/control arm. No LGSSM recipe or weight is transferred as a
default.

Required artifacts: per-cell force/training/tuning/sampling/cost artifacts plus
a joint close record that never merges the two posterior identities.

Required checks/tests/reviews:

- per-cell target replay and deterministic endpoint repetition;
- target-specific force training and no cross-filter artifact reuse;
- tuned raw-coordinate plain HMC plus zero-residual and true-gradient chart
  baselines; matching preserved plain-HMC evidence may be reused after replay;
- modern sampler health and six physical-parameter truth-tail tables;
- retain the noncentral-truth label for the historical fixture;
- separate UKF and SGQF cost/validity decisions;
- apply the descriptive performance screen separately in each cell; a valid
  but slower cell remains validity-confirmed with performance not demonstrated.
- require a separate value-only UKF/SGQF endpoint and parity against each
  complete transformed value/score target; no unused true filtering gradient
  may be hidden in endpoint cost.

Evidence contract: each pass is exact only for the named deterministic filter
posterior and one fixture. Filter ranking is not tested.

Forbidden claims/actions: no UKF/SGQF equivalence, latent-model exactness,
calibration, cross-cell force reuse, or pooled pass.

Exact P5 handoff: both cells are classified; any local failure has consumed its
two repair attempts or has a precise re-entry rung; the shared kernel remains
valid; P5 budget is available.

Stop conditions: shared contamination or program budget veto. One cell failure
does not stop the other or P5.

Phase-end duties: run checks; write P4 result; refresh P5; review P5; continue
if no real blocker.

Skeptical audit, refreshed 2026-07-17: passed conditionally on model-specific
value-only parity. The two cells remain independent. Preserved target-matched
NeuTra coordinate archives may supply disjoint supervision slices after hash
replay, while all force weights and tuning/retained seeds remain cell-specific.
Loss cannot promote a force; zero residual may remain the representative valid
arm if learning supplies no downstream benefit.
