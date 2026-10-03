# HMC continuation with supplied funnel whitening

## Revised question and scope

The owner's September 22 clarification makes a supplied frozen whitening or
partial-whitening map the premise of positive funnel tuning tests. The question
is whether the existing public fixed-transport tuner correctly searches and
verifies kernels in those coordinates. Learning a useful map is a separate
upstream task. Failure of the untransformed centered funnel is retained as a
stress-test outcome, not an outstanding requirement to make ordinary tuning
succeed. No finite epsilon/L search guarantees success on arbitrary geometry.

Baseline is commit `01d67ec41`, including the bootstrap probability and public
ESS corrections. The running M21/M22 confirmation uses `f9c86f41a`; its immutable
source and denominators remain unchanged. This continuation adds current-source
integrations rather than relabeling those older experiments.

## Revised gaps and closure program

| Gap | Closure evidence and next action |
| --- | --- |
| Supplied-map tuning integration | Check exact and partial maps, target/score/Jacobian, same model starts, frozen identity, fresh per-pair verification, complete retention, export/reload and posterior-coordinate mapping. Execute the bounded matrix below. Raw centered failure is a stress outcome. |
| Residual geometry and finite search | Record map quality analytically; distinguish no viable measured pair, finite-budget exhaustion and lost verification. For a partial map that fails, inspect numerical and directional evidence before one bounded explicit-grid repair in the same coordinates. Map improvement is upstream work, not a tuning default. |
| Posterior precision and global exploration | Preserve verified siblings. Diagnose requested units, tail variability, dependence and mode crossings separately. Existing M21 Gaussian/rotated/SSM controls continue; mixture occupancy remains a posterior gap, never an epsilon/L admission condition. |
| Warmup, MCSE and data-dependent stopping calibration | Finish the fixed M21 inventory; compare stopped and independent fixed-count arms including caps and missing estimates. Existing lugsail bandwidth failures remain open. Resolve a separately calibrated estimator/bandwidth study before changing defaults. |
| Validation false-positive rate and defect power | The fixed M22 null/no-op confirmation now passes its declared screen; see the September 22 null result. This closes that frozen-Gaussian size-precision cell only. Repeated subtle full-fit power still needs a measured-cost redesign; preserve the under-budgeted finding instead of shrinking the denominator until the study becomes uninformative. |
| Exact MacroFinance case and bounded recovery | Fresh source/target-qualified 22/180/22 preparation, unchanged verification and posterior assessment remain necessary. Exact inputs/reference are absent here. XLA exception attribution remains unsupported; do not classify arbitrary errors as rejected proposals. |
| Current-source/backend regression and maintenance | Run the new integrations on the repaired source, migrate the useful start-coordinate invariant into tests, and rebuild the official book. GPU evidence is conditional on trusted permitted capacity. Profile before broad refactoring. |
| Upstream learned-map quality (separate) | M23 training engineering is tested; target-specific trained-map quality remains unvalidated. It is not a prerequisite to the supplied-map tuning tests or a tuning-repair completion criterion. |

After each phase: reconcile charged work, classify implementation/geometry/
posterior/evidence failure, perform a justified bounded repair, record results,
and refresh the next phase. Candidate rejection does not stop other work.

## Maps and independent mathematical comparator

The physical model is v ~ N(0,sigma^2), x_i | v ~ N(0,exp(v)) for two
children, with inherited sigma=3. For the exact control, use the existing
noncentered adapter (v,u), x_i=exp(v/2)u_i, and a supported frozen affine map
v=sigma*z_0, u_i=z_i. The total model map has log determinant log(sigma)+v.
The conditional normal log-scale terms cancel v, leaving three independent
standard normal coordinates. The analytic noncentered chart and affine map
are both checked; this is not a learned map or a new public codec.

For partial maps use the existing dense-IAF codec with one affine component
v=sigma*z_0 and a smooth autoregressive component

    s(v) = c*tanh(a*v/(2*c)),   x_i = exp(s(v))*z_i.

Its log determinant is log(sigma)+2*s(v). The exact transformed conditional
standard deviation is exp(v/2-s(v)); this residual explicitly measures the
remaining funnel variation. A two-unit tanh hidden layer with existing masks
and identity log-scale transform represents s exactly. The second hidden unit
is unused. No ReLU kink, approximation to the derivative, new activation,
arbitrary callback or runtime adaptation is introduced.

Declare c=2*sigma=6 as a convenience saturation hypothesis. Test a=1 (locally
whitened, residual tails) and a=0.5 (half of the local log-scale correction).
These are fixed sensitivity controls, not trained settings or recommended
defaults. Check v from -9 to 9, including zero, with nonzero child coordinates;
the +/-3-sigma range is an algebraic diagnostic domain, not a support truncation.
Invertibility and the density identity hold for all finite inputs within
floating-point range. The tails outside that diagnostic range remain part of
the target and may cause valid numerical rejections.

Inspect both live transport pullbacks and independent autodiff of the derived
latent density. Confirm that omitting the log Jacobian changes the density and
score where nonconstant. Independent model truth/reference cannot initialize
or choose a tuning candidate.

## Engineering repair and tests

Inspection found that `execute_pipeline` passes its base/model-adapter start
bank directly as `initial_position` to the fixed-transport tuner, whose public
contract requires latent starts. The target law is unchanged, but previous
cross-map start comparisons were not matched. Convert starts using the loaded
map inverse (explicit affine inverse for its codec), check the forward
roundtrip, and preserve model/adapter/latent banks in a start record. Existing
completed runs remain historical evidence of their actual starts; do not
rewrite their artifacts. Direct public API semantics stay unchanged.

Unit tests cover inverse/forward/Jacobian/score identities, zero and tail
coordinates, invalid map parameters, distinct artifact identities and matched
physical starts. Absolute and relative tolerances of 2e-11 are engineering
margins for these float64 analytic identities, not posterior-error thresholds
or a lower-precision transport default. Independent formulas and tail inputs
check whether those margins mask a wrong map. Public integrations must use real measurements and independent
verification, retain every passing receipt, reload each selected member and
exclude warmup. No mocks, widened acceptance bands, R-hat gates or skipped
failed positive cases can establish this result. Re-run existing affine/dense
route integrations and candidate/guide contracts after the harness edit.

## Numerical execution and evidence contract

Execute three maps with two fixed independent seeds (2026092251, 2026092252),
chosen as fresh development identifiers. Six fits are mechanism/applicability
evidence, not a statistical coverage or method-ranking study. Use broad
L=(3,5,9,13,18,25), epsilon warm start .5, 128 measurement and 128 verification
draws, and the existing acceptance policy, pilot, one refinement round and
finite evidence rungs. These are inherited M17 integration settings, not
target-specific optimum claims. Select the first verified identity before
posterior assessment; keep every sibling. Preserve the M17 .05 absolute
mean/median precision request, 2000 warmup minimum, 1000 recent window,
500-transition chunks and 10000 per-chain caps. Posterior caps remain failures
of that requested precision, not tuning failures. No passing posterior is
required to establish a correctly verified tuning set.

Primary engineering criteria: exact target/map/start identities, independent
stages, complete candidate retention, checked replay, model-coordinate mapping
and explicit outcome classification. Positive tuning usability additionally
requires a verified member on each exact-control fit. Partial-map outcomes
measure applicability at declared residual curvature; an empty set triggers
inspection and at most one predeclared explicit-grid follow-up, not a map or
threshold change. Search correction is justified only by a demonstrated
missed mechanism, not by the expectation that every partial map must pass.
Numerical corruption, wrong law/score/Jacobian, broken provenance or missing
required telemetry vetoes the affected run and triggers a repair. Acceptance,
candidate counts, R-hat/ESS/MCSE and times are descriptive in this small matrix;
posterior criteria apply only to posterior results. No universal robustness,
learned-map quality, ranking, nominal coverage or new default is concluded.

Reserve 3600 CPU worker-seconds for tests, the six-fit CPU development matrix
and bounded local repair; 3600 GPU worker-seconds for the same matrix if capacity
is permitted. These engineering ceilings use the M20 measured complete-tranche
cost (1397.76 seconds) as a rough planning comparator, not a runtime promise.
Measure the first fit before admitting the remaining five. A per-fit 360-second
limit bounds the initial CPU inventory to 2160 seconds; the rest covers tests
and local repair. All failures count. The existing 48 CPU/24 GPU hour ceiling
and the 79800-second live confirmation reservation remain intact. Move up to
1800 uncommitted CPU seconds from M23 to M20 if its old one-hour allocation
would otherwise bind; learned training is separate from this tuning tranche.

At most one additional single-thread diagnostic worker may accompany the two
existing confirmation workers during this bounded tranche. This temporary
exception ends when the tranche finishes; no runtime comparison is made.
CPU reference runs deliberately set CUDA_VISIBLE_DEVICES=-1 before import.
GPU runs require trusted selection, TF_FORCE_GPU_ALLOW_GROWTH=true, verified
memory growth and XLA. No device contention, package changes or training is
authorized by this note. Missing GPU capacity leaves only the GPU cell open.

Commands: the existing tfgpu Python runs the focused pytest modules and the
new `docs/benchmarks/run_hmc_supplied_funnel_maps_2026_09_22.py` with explicit
`--output`, `--device`, `--seed`, `--map` and `--seconds`. Fresh output root:
`docs/plans/artifacts/hmc-repair-master-2026-09-16/m20-r2/`. Preserve source
hashes, Git commit/diff, command/environment, device and thread policy, map
payloads, seeds, target identity, start records, candidate receipts, posterior
chunks, wall time and result. Write the result note alongside this plan and
update the master and progress record. Build `docs/main.tex` in a fresh output
directory and inspect the changed rendered pages.

## Skeptical review before execution

The old success expectation conflated fixing geometry with tuning a supplied
geometry. This plan corrects that baseline and keeps unwhitened outcomes visible.
Assuming an ideal learned map without checking its density would make the test
meaningless; the maps here have explicit independently checkable laws and frozen
codec reconstruction. An exact-only Gaussianized test would miss residual
curvature; the two partial maps retain it. Starts in the wrong coordinates
would confound comparison; repair their inverse mapping first. The full control
uses an analytic chart, so its result cannot establish that a learner discovers
that chart. Short successful chains do not establish burn-in or model-moment
precision, especially for the funnel's lognormal scale mixture. No posterior
criterion can remove tuning members. Fresh source provenance and immutable old
confirmation avoid attributing old results to new code. Fixed limits, captured
failures and the explicit GPU capacity condition prevent an unbounded search
for favorable results. These checks resolve the material design flaws; proceed
with algebraic and public-route tests, then the measured-cost development run.

## First phase result and bounded next phase

All 28 focused checks passed. Both exact-map fits retained verified members
(six and eight), while all four partial-map native searches returned empty
sets. In the first a=1 fit, epsilon .5 had high finite acceptance; L=5 at
epsilon 1 measured as compatible but fresh verification became nonfinite.
Other doubled steps also became nonfinite. For a=.5, epsilon .5 produced
compatible measurements at L=3/5 but verification either disagreed or became
nonfinite. These are preserved finite-search and residual-geometry outcomes;
they do not invalidate the map algebra or justify assigning acceptance to an
invalid endpoint. Both exact-map selected posteriors hit the retained precision
cap, without chunk hard vetoes. A launcher path typo consumed .0324 seconds
before any numerical run and was repaired in a new attempt.

Execute the previously allowed single explicit-grid follow-up for each partial
map/seed. Preserve map, model starts, policy, L grid, counts, posterior settings
and caps. Use epsilon=(.5,.625,.75,.875,1,1.125,1.25) for a=1 and
(.25,.375,.5,.625,.75) for a=.5, at every declared L. These are diagnostic
subdivisions around the observed finite/nonfinite region, not inferred safe
steps or calibrated defaults. Disable per-L pilots so every declared pair
gets its own measurement; enable two finite refinement rounds with the existing
failed-interval exploration option. The proposal-only rejected endpoint never
supplies an acceptance direction. Keep max_candidates=100, total work units=300
and the existing per-fit 360-second external bound. A distinct design ID gives
fresh streams; this is new development evidence, not reused qualification.
Do not add another search after this follow-up solely because it fails.

The original six fits cost 358.80 CPU seconds plus the path failure, leaving
ample room in the 3600-second tranche for four bounded follow-ups and remaining
contract tests. Skeptical review: a denser grid plus optional interval
exploration is a combined search hypothesis, so it cannot identify which
feature causes a difference. A successful member still requires fresh
verification and does not certify residual tails or posterior precision. A
failed partial-map grid leaves geometry applicability open, rather than an
unconditional obligation to change the tuner. No threshold, map or target
change is part of this continuation.

## Final regression finding and reviewed local repair

The final four-module regression returned 62 passes and one failure. The
supplied-map integration, including the actual checkpoint start bank, passed.
The existing ordinary positive-handoff test used the 12-transition smoke
schedule. Its four final-window rows contained one repeated MH state; the
start-bank rule correctly rejected the insufficient distinct states. Earlier
windows provided enough distinct points, but their shadow diagnostic is not
authority to bypass the final-window rule. Preserve the complete failed
preparation record under `m20-r2/tests-r2/`.

Change this positive integration test to the existing standard preparation
preset (128 warmup transitions for this dimension), with the same target,
initial position, root seed, public route, start-bank rule and assertions.
This is an inherited test budget, not a new runtime default or a convergence
claim. Skeptical review: choosing a favorable seed or weakening dispersion
would hide the failure. Giving the positive-path fixture the standard budget
instead tests the intended public preparation handoff; existing dedicated
start-bank failure tests preserve its negative behavior. Rerun this changed
test and the focused start-bank failure tests only, since all other final
checks passed. Record the outcome and include both attempts in the tranche.

## Execution closeout

The [terminal result](bayesfilter-hmc-supplied-whitening-result-2026-09-22.md)
records all ten fits, the bounded follow-up, 80 distinct passing regression
checks, the built/inspected guide, preserved failed attempts and the revised
next phase. The selected partial-map posteriors failed numerical health;
exact-map posteriors missed model-mean precision at the unchanged cap. The
engineering cell is closed, those posterior and calibration gaps remain open,
and the existing frozen M21 confirmation continues. Tranche charge is
1568.334463 CPU seconds and zero GPU seconds. The extra-worker exception ends
here. No further search or default promotion follows from these small-sample
development results.
