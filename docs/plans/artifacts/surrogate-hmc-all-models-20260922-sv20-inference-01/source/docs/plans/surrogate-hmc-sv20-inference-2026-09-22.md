# Actual SV20 first B/R surrogate-HMC test

## Research intent and evidence contract

Test whether endpoint-corrected HMC using the full canonical analytical DCTR
position-only field recovers the actual SV20 parameter posterior. Preserve the
original attempt1 claim data, stationary initial law, unit process variance,
raw Normal(0,beta² exp(x)) observations, independent Uniform(.1,.9) gamma/beta
priors, probit chart and N=K=1024. Use the disjointly selected multiplier0.1.
The force is an imperfect deterministic proposal field; agreement with the
endpoint gradient is not required. Force settings and frozen particle banks
must not adapt during sampling.

B uses the qualified FP64 1024-interval[-16,16] numerical model endpoint,
with explicit numerical failure for predicted domain loss above1e-9. R is the
independent full-prior posterior reference, including1280[-20,20] refinement.
Both approximate the same actual-SV law; neither is an exact model oracle.
Finite-DCTR C/D, D/R and C/R remain NOT_RUN and separate obligations.

The primary criterion is B/R equivalence of all13 frozen functions under the
shared `assess_equivalence` procedure and the reference's refinement allowance:
standardized gamma/beta means, centered seconds and cross moment (margin.1),
physical marginal CDFs at R's10/50/90 percentiles (.02), and posterior predictive
probabilities for |y21|<=.25 and1 (.03). Use four chains and only cumulative
1,000/2,000 retained draws per chain. Allocate simultaneous error across13
functions, two looks and four comparison families. Rhat, bulk/tail ESS,
batch-MCSE sensitivity and batch correlation checks must pass before declaring
equivalence or discrepancy. An invalid uncertainty screen is INCONCLUSIVE.
Stop after a valid EQUIVALENT or DISCREPANCY; otherwise finish the second look.
Never repeat a closed look or relax the MCSE rule after observing it.

Public `tune_hmc_kernel` with the typed independent force/endpoint binding is
the sole tuning authority. Its nomination,250 discarded warmup and1,000 fresh
verification draws are excluded from retained inference. Failed tuning rejects
the current attempt, not surrogate HMC. Every retained chunk must pass finite
state/target/log-acceptance, original force status, all-chain movement and
absolute energy-error<=1,000 checks. Source/data drift, invalid archives,
required reference failure and exhausted compute are continuation vetoes.
Poor mixing or insufficient uncertainty at the first look permits the declared
second look. Acceptance, force/reference differences, timing and observed ESS
without comparative uncertainty are explanatory only.

Heuristic adversaries are the prior-center conditional predictor (ignore
parameter uncertainty), prior predictive integration (retain uncertainty without
learning from observations), and Gaussian/Kalman moment approximation (test
whether cheap Gaussian inference suffices), evaluated separately for low/high
absolute returns. These are NOT_RUN; usefulness, efficiency, superiority and
default promotion remain blocked. Agreement of13 functions does not prove
equality of the full distributions or validate the remaining SV variants.

## Assumptions, defaults and skeptical audit

| Choice | Provenance and justification | Failure mode and early check | Status |
|---|---|---|---|
| Actual-SV law, chart, reference and full DCTR | Qualified likelihood/posterior and17 endpoint checks | Wrong law, truncated mass or wiring fork; prior binding, per-call validity and reference comparison | Frozen scope |
| One leapfrog step | The complete measured forecast makes a first L1-only test affordable; endpoint correction remains the tested mechanism | Random-walk-like movement; fresh verification and retained diagnostics can veto | Restricted proposal hypothesis, no efficiency claim |
| Starts(-2,-2),(-.7,.7),(.7,-.7),(2,2) in probit coordinates | Standard-normal induced prior covers both parameter directions | Unvisited modes; independent R and functionals veto | Target-specific starting hypothesis |
| Initial identity metric and epsilon.1; mass windows32/64;128 adaptation and64 nomination verification | Standard-normal prior supplies scale; independent adaptation followed by fresh verification | Short adaptation nominates poor geometry; public-tuner gates | Warm start only |
| Target acceptance.7, covariance shrinkage.1, jitter/floor1e-9 | Existing tuner with an identity prior scale; a tiny positive floor bounds factorization risk and realized covariance checks are retained | Geometry can still be poor; no covariance setting is promoted by this run | Numerical candidate with recorded rationale |
| Two retained looks, fixed diagnostic rules | Existing master evidence contract; reserved complete cost | Underpowered intervals or batch correlation failure | Bounded inference test, not convergence guarantee |
| GPU/XLA, FP32/TF32 force, FP64 endpoint/integrator | Repository execution policy and qualification | Allocation or precision failure; growth and health checks | Required configuration |

Skeptical audit passed before execution: R and B share the actual raw-return
law and claim data; the proposal field is the full canonical implementation;
controls use independent calibration/validation; all posterior functions and
uncertainty rules are frozen. The17 endpoint checks and15 focused tests passed.
The complete forecast counts3586 four-chain transitions and25632-row predictive
batches. Cache functionals once per retained chunk; do not recompute earlier
draws at each look. Reference refinement enters the equivalence allowance.
No proxy loss, timing or short-chain acceptance can promote inference.

Pre-mortem: L1 may pass a short tuning check but mix poorly; retained MCSE and
R detect that. Sparse endpoint probes might miss a tail issue; the full-prior
reference and per-call domain guard remain necessary. A healthy numerical run
may still end INCONCLUSIVE, which must be preserved as such.

Final implementation audit before launch: the master dispatch reaches the
public verified tuner with independently typed force and endpoint bindings;
retained sampling reloads the issued tuning artifact. The thirteen functionals
use physical gamma/beta and the qualified reference's final expectations and
refinement allowances. Archived warmup and fresh verification are excluded.
All57 distinct focused checks pass, including corruption, closed-look and
dispatch checks. The first GPU metrics smoke failed before sampling because
XLA gave a dynamic slice the full1024-row bound. A gather from fixed32-row
batches repairs the shape without changing numerical formulas; the focused
regression and fresh GPU smoke02 pass, with predictive difference0 against a
separate four-row evaluator. Preserved tests/smokes cost33.916205 CPU seconds,
within the60-second check reservation. Global charge before launch is
16702.306183 CPU/14949.240531 GPU; SV family remaining12341.179354/13891.018900.
No tuning or retained look has been consumed. Source/data validity, stop
conditions and the complete cost were rechecked; no material blocker remains.

## Allocation and command

After qualification, completed global use is16,668.389978 CPU and14,933.899787
GPU seconds. Transfer11,200 unused CPU seconds from P4 to P3: P3 becomes29,500,
P4 becomes3,200 and P5 remains3,300. Increase the cumulative SV family ceiling
from1,800 to13,000 CPU and from7,200 to14,400 GPU seconds using existing P3 GPU
capacity. The global54,000 CPU/226,800 GPU caps are unchanged. Other ready
families have received their initial preparation turns; SV now has all required
B/R prerequisites. This allocation funds actual inference after qualification.

Reserve12,000 CPU/12,600 GPU-wall seconds for exactly one public-tuned B attempt,
plus60 CPU for implementation checks and30 for orchestration. The measured
1.5-factor forecast11,655.452986/11,661.125201 fits. All prior SV costs, failed
attempts and tests count against the family ceiling. No automatic new attempt
or additional statistical look is authorized by an inconclusive result.

Execute `scripts/run_surrogate_hmc_program.py --stage P3-SV20-inference
--scope SV20 --output docs/plans/artifacts/surrogate-hmc-all-models-20260922-sv20-inference-01`
using `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, trusted GPU1 UUID
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, growth, two CPU threads and XLA.
Seeds: tuning[20260922,35000], fresh verification[20260922,35001], retained
chunks[20260922,36000+i], i=0..7. Preserve exact commands, data/source hashes,
source snapshot, tuning manifests, warmup/verification traces, eight retained
archives at most, looks, report and accounting. Only one serious GPU job runs.

Record outcome, decision/inference-status tables, alternative explanations and
remaining budget in the result note. Keep engineering correctness, sampler
validity and scientific interpretation distinct.
