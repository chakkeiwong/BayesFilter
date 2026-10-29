# Shared filter component-cost diagnostic

Question: which shared calculations account for expensive N4032 analytical
force evaluations? The prior GPU profile contained no timed kernels; it cannot
support attribution. This is a diagnostic phase of the authorized master
repair, after current SIR/range tuning. Budget at most1500 GPU seconds/five
workers, charged to the existing36000-second ledger.

Compare the actual one-observation shared factory with a staged invocation of
its existing proposal-map/draw, density/weight and reset primitives. All paths
use the same observations, theta, fixed normals/uniforms, controls, N4032,
K2016, batch2 and precision policy. No reduced numerical kernel is introduced.
The composed result must agree in value, every score, particles and particle
tangents to1% of max(1,abs(reference)); any invalid/nonfinite result vetoes
timing interpretation. Preserve exact and scaled differences. Three warmed
timings are descriptive only, with compilation reported separately.

The diagnostic uses the first observation of each requested scope and two
nominal/nearby parameter rows. This isolates an actual supported step but
cannot establish whole-horizon fractions or posterior-region cost. Staging
can alter XLA fusion/layout and creates intermediate materialization; report
its total relative to the fused step. Large disagreement in timing limits
attribution, not numerical correctness. Each primitive retains explicit
stable signatures and XLA. Return finite/valid/full-step checks and existing
reset diagnostics. No learned component, tuning authority or HMC promotion.

The candidate is engineering instrumentation, not a changed method.
Comparator is the actual unchanged factory on identical data; exact Kalman,
BPF or other scientific baselines answer a different question. They remain
required for later algorithm/HMC quality claims. No statistical speed ranking
or precision-default change follows. Defaults are owner N4032/FP32/TF32off;
optional mixed arithmetic must be labeled. Existing controls are own-scope
warm starts, frozen for this cost probe. First-step data is a convenience
choice exposed by the full-horizon limitation above.

Pre-mortem: a fast staged graph may simply compile differently; a missing
score/reset term could make it falsely fast; a dead-code-eliminated output
could hide work. Compare all returned finite-filter outputs, materialize all
diagnostic stage results, and keep the fused comparison. CPU/XLA tiny-fixture
parity and actual master flag wiring precede GPU launch. Preserve current
running numerical sources. Audit passes for localization only. Next action
depends on the measured expensive primitive; do not optimize an assumed
bottleneck. Store command, source hashes, environment, device/memory policy,
seeds, wall time and raw timings in a fresh versioned result directory.

## Pre-execution audit and checks, 2026-10-12

The stage implementation calls the existing shared functions and uses the
actual Models callback property to determine constant observation Jacobians.
All eleven deliberate CPU/XLA checks passed across five models and both
precision policies (190.83 seconds). All93 master command/orchestration
checks passed. Logs: component-cost-regression.log and
master-component-wiring-regression.log in this directory. A final provenance
assertion verifies actual controls, design seed, parameter rows and input
checksums. The old profiler failed at cuptiGetTimestamp with error999 and
collected zero callbacks/events; it is not evidence about expensive kernels.

The master flag is --component-cost, with --runtime-precision only when the
optional mixed arithmetic scope is requested. It cannot be combined with
tuning, mechanics, selection authority or other diagnostic modes. Each run
uses a fresh versioned output root and the existing cumulative ledger.
For each model MODEL, the exact command form is:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py run --model MODEL --component-cost --output docs/plans/artifacts/surrogate-hmc-component-MODEL-20261012-01 --budget-seconds 300 --per-model-timeout 300 --campaign-ledger docs/plans/artifacts/surrogate-hmc-repair-continuation-20261012-01/budget.json
Any mixed arithmetic option is recorded in the actual command. These commands
are part of the master repair, after pending SIR and range tuning. No
additional campaign authorization is required.
