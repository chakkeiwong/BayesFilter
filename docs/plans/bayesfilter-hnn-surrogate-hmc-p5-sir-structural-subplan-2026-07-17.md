# P5 Subplan: Spatial SIR SGQF And Structural UKF

Phase objective: test the corrected kernel on the admitted SIR-SGQF posterior
and the Chapter 18b structural-UKF posterior, preserving structural semantics.

Entry conditions: P4 closes; both target/chart identities replay; structural
deterministic identity and no-artificial-noise checks pass before training;
28-GPU-hour ceiling is frozen.

Refreshed entry evidence, 2026-07-18: P4 passed independently in both
predator-prey cells, but it also showed that a strong frozen NeuTra chart can
make zero residual sufficient. P5 therefore retains zero residual as a
co-primary validity/control arm and treats learning as a target-specific
hypothesis. No P4 architecture, learning rate, force weight, or tuned kernel is
transferred as a default.

Required artifacts: independent SIR and structural training/tuning/sampling/
truth/cost records, structural invariant trace, and cell decisions.

Required checks/tests/reviews:

- SIR three-parameter and structural five-parameter truth-tail tables;
- modern rank/folded R-hat, ESS, energy, divergence, and target-status gates;
- structural identity `k_t-phi k_(t-1)-gamma m_t^2=0` on all required points;
- no process noise injected into the deterministic structural coordinate;
- disclose the historical structural bulk-ESS owner adjudication, but apply the
  current frozen P5 threshold prospectively; a post-result change receives only
  the master's qualified posthoc label;
- matched tuned raw-coordinate plain HMC, zero-residual, and true-gradient chart
  baselines; matching preserved plain-HMC evidence may be reused after replay;
- apply the descriptive performance screen separately in each cell.
- add separate value-only SIR-SGQF and structural-UKF endpoint programs and
  prove parity against the complete transformed value/score target before any
  training or corrected chain; endpoint evaluation must not allocate unused
  parameter derivatives;
- preserve the historical target/transport execution identity while recording
  the source-provenance refresh caused only by adding endpoint functions.

Evidence contract: passes concern the named SGQF/UKF posteriors and fixtures.
The structural result also requires the deterministic invariant; sampler
diagnostics cannot compensate for structural corruption.

Forbidden claims/actions: no latent-model exactness, calibration, structural
noise relaxation, silent ESS threshold change, or pooling the two cells.

Exact P6 handoff: Tier A is cell-complete; shared kernel remains valid; Tier B
standard-target adapters and cost ceiling are refreshed. P6 may proceed even if
one P5 cell is honestly blocked.

Stop conditions: structural target invalidity blocks STR-UKF only unless it
reveals shared contamination. Shared kernel or total-budget veto stops program.

Phase-end duties: run checks; write P5/Tier A close result; refresh P6; review
P6; continue if no real blocker.

Skeptical audit, refreshed 2026-07-18: passed conditionally on scalar endpoint
parity and the structural invariant/no-noise gate. Existing target-matched
frozen transports and sample archives may be reused after hash replay. A local
candidate failure is not a continuation veto for the independent cell or P6.
