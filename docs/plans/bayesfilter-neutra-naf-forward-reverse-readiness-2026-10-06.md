# NAF study default and forward/reverse readiness

The monograph now records the completed scalar-nonlinearity study, and full
Huang DSF/NAF with the author cMADE conditioner is the default for this NeuTra
training study. The refreshed master can start measured native-teacher
calibration. The full scientific procedure has not yet run.

The study supports a directional density-fitting effect under the paired
experiment: all four confirmation contrasts have eight positive pairs and
Bonferroni-adjusted sign-test p=.015625. Thirty of 32 nonlinear confirmation
fits passed the distribution screen; two three-mode fits remain rejected.
The transfer control passed 8/8 nonlinear and 1/8 matched affine maps. Counts,
tail differences, timing and the two-seed transfer controls are descriptive,
not a general ranking of posterior samplers. The new default is the owner's
architecture choice for this work, not a claim of universal IAF impossibility,
q20 readiness, TF32 adequacy or reliable approximate-teacher training.

## Documents and implementation

- Scientific result: [attribution results](bayesfilter-neutra-nonlinearity-attribution-results-2026-10-06.md).
- Manuscript section: [scalar nonlinearity](../chapters/ch26f_neutra_nonlinearity_results.tex), included through `ch26f_neutra_controlled_training.tex` into `main.tex`.
- Compiled monograph: [PDF](artifacts/neutra-naf-forward-reverse-2026-10-06/monograph/main.pdf), 654 pages. The new proof and result table were inspected in the rendered PDF. Existing warnings elsewhere remain.
- Active protocol: [forward/reverse plan](bayesfilter-neutra-naf-forward-reverse-master-2026-10-06.md).
- Executable master: `scripts/run_neutra_scientific_campaign_master.py`; phase queue: `scripts/neutra_forward_reverse_campaign.py`; numerical route: `bayesfilter/testing/neutra_forward_reverse.py`.
- Machine-readable next phase: [program](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-program.json).

The default constructor now selects three stages, two width-64 hidden layers,
four-component scalar DSF and batch 64. These numerical values remain the
initial study recipe, subject to calibration. Generic artifact deserialization
and q20 consumers retain their meanings; repaired author IAF is an explicit
comparator. The scoped directive is recorded in `AGENTS.md`, `CLAUDE.md` and
`docs/reference/neutra-implementation.md`.

## What the refreshed program tests

The queue performs mode search and constructs a normalized defensive Laplace
proposal using only the target density, derivatives and dimension. Independent
AIS/SMC populations run on CPU with GPUs hidden. Four training and four
validation populations form separate banks. Within each population its weights
are normalized, then each complete population receives weight 1/4 in the pooled
bank. Exact draws are reserved for evaluator files, never native training.

NAF forward training minimizes the resulting finite weighted cross entropy.
Each pure reverse-KL branch starts from the same saved forward parameters with
fresh Adam and fresh base noise. Independent teacher checks and both endpoint
distribution screens must pass; a lower reverse objective does not override
coverage loss. The forward endpoint, reverse endpoints and every 1,000-point
probe remain preserved.

Calibration uses development geometries 9101/9201 and fitting seeds 601/607/613.
The first complete passing recipe is frozen without ranking passing methods.
All six fixed warped/unwarped fitting cases must then pass before testing four
reserved random geometries with three fitting seeds each. A fixed-stage failure
rejects that recipe, preserves the random holdouts and calls for a development
repair; it is not evidence against NAF as a research direction.

SMC and AIS are executable teachers. FAB remains deferred for auxiliary-target
tail eligibility, Gabrié for the complete adaptive controller, and AFT/CRAFT
for controller equivalence. These are incomplete routes, not algorithm-failure
verdicts. This program does not pretend to finish the six-method comparison.

## Engineering validation and repair

The initial 42-check suite passed. The native sampling regression and queue
tests passed seven checks after the tracing repair. Final recovery/controller
checks passed 22 tests, including terminal idle resume, failed-candidate
continuation, fixed-before-random gating, larger-teacher timing isolation,
budget refusal and settlement of pending verification costs. These sets overlap.
Logs are in the readiness `check-r1` directory and the campaign's
`forward-reverse-regression-check-r1` / `forward-reverse-recovery-check-r3`.

The first CPU smoke failed after 13.99 wall seconds (25.35 CPU-core seconds)
while concurrent populations constructed derivative graphs of a shared
target/proposal. No GPU training began. Tracing the stateless kernel once before
concurrent execution repaired the defect; the failed attempt is preserved.

The successful smoke used CPU generation followed by physical GPU 2, TensorFlow
GPU/XLA, FP64, batch four and the canonical configured NAF. Its four forward
and four reverse updates were finite. Forward and both reverse checkpoints
reloaded, and all three probes had exactly 1,000 finite rows. Memory growth
was verified before device initialization. Native teacher cost was 12.83 wall /
27.61 CPU-core seconds; GPU fit cost was 154.37 wall / 192.76 CPU-core seconds.
These timings include setup/compilation and do not estimate full training cost.
The [smoke record](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-smoke.json)
links the complete worker manifests and preserved numerical artifacts.

Final review also repaired stale recovery behavior and resource reservations.
A failed/interrupted smoke now resumes its own route, and a larger teacher
profile cannot inherit the smaller profile's timing. The device-free plan
refresh settles pending test charges. Each scientific worker still gets a
fresh source snapshot/output directory and the existing shared budget lock.

## Decision and limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Use NAF by default for this study | Owner direction plus checked attribution evidence | Two nonlinear fits remain rejected | Reliability beyond the exact-teacher study | Test approximate teacher and objective switch | Universal NAF superiority |
| Master ready to begin calibration | Native CPU to GPU/XLA call chain and recovery tests pass | Initial tracing failure repaired; no active worker | Full-scale teacher and fit costs | Run the first complete development pair | Full campaign affordable or scientifically successful |
| Keep random geometries reserved | Frozen recipe and all fixed cases must pass first | Failed fixed stage blocks exposure | Forward/RKL coverage with approximate data | Complete calibration and fixed stage | Generalization or q20 posterior validity |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Two attribution fits fail their distribution screen; no failure is overridden by lower loss. Readiness smoke has no unresolved numerical/integrity veto. |
| Statistically supported ranking | The predeclared paired study supports directional density benefit; no new native-teacher or posterior ranking is established. |
| Descriptive-only differences | Mean KL benefits, counts, tails, control timings and tiny smoke metrics. |
| Default readiness | NAF is the scoped owner default; the complete approximate-FKL/RKL recipe is not validated. |
| Next evidence needed | Independent native teacher screens, target-specific calibration, fixed-target success, then untouched geometry transfer. |

The strongest alternative explanation for later failure is inaccurate or
incomplete teacher coverage, or collapse during RKL, despite adequate scalar
expressivity. The smoke cannot distinguish these explanations; separate
teacher and frozen-map assessments in the actual calibration are required.
No long-run runtime or success forecast is warranted from the tiny check.

All costs are settled. Remaining allocation is 38,030.72 GPU-process seconds
(10.56 hours) and 32,834.95 CPU-core seconds (9.12 hours). Initial ceilings and
measured reservations are stated in the active plan; the master must report
under-budgeted if a complete next stage cannot fit. No full calibration or
scientific holdout run has launched during this refresh.

Exact commands:

```bash
bash scripts/run_neutra_scientific_campaign.sh forward-reverse-plan
bash scripts/run_neutra_scientific_campaign.sh forward-reverse
bash scripts/run_neutra_scientific_campaign.sh resume
```

The first refreshes status without a numerical worker. The second starts or
continues the new study; the third dispatches to the prepared/active study.
GPU execution remains subject to normal trusted tool permissions.
