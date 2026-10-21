# AR2 finite-likelihood closure stage — 2026-09-23

This is the next finite-target comparison in the existing surrogate-HMC master.
B/R has already tested the canonical DCTR proposal force with model-posterior
endpoint correction. The present question is different: does the deterministic
N64 DCTR likelihood itself give a posterior close to the AR2 model posterior?
D denotes numerical integration of that finite likelihood times the declared
prior; R is the already qualified independent Kalman/grid posterior. Later C/D
tests sampling of D, and C/R combines the approximation and sampling questions.
B/R and A/R retain their archived outcomes and receive no additional looks.

## First bounded action and evidence contract

Before allocating a dense D reference, run ar2-finite-preflight. Use the same
claim data, N64 controls, frozen particle/noise/reset design and batch-four
consumer used by B. The finite endpoint must resolve to that consumer's actual
canonical value. No new filter, artificial process noise, numerical support
truncation, force substitute or DCTR control selection is introduced.

The first action screens the domain and measures complete canonical evaluation
cost. Use all 49 combinations of prior-standardized coordinates
{-6,-4,-2,0,2,4,6}, plus nine combinations of {-2,0,2} posterior SDs around the
independent reference mean. The prior grid exposes central and tail numerical
failures; the reference-centered grid checks the part most relevant to the
posterior. Reference mean and scale specify probes, not a parameter fit or
control selection. No observed D/R discrepancy is used to tune the method.

Primary preflight criteria are valid canonical finite endpoint values on every
probe, finite exact comparator values, reproducible results and invariance to
batch row permutation. Keep value validity and force validity separate.
A numerical endpoint failure vetoes dense-reference launch under this design;
it does not define zero probability or a support boundary. It triggers
localization, a scoped numerical repair or a justified integration-domain
analysis. That failure does not reject B/R or the surrogate-force direction.

Compare both endpoints at the same points and save raw log-density differences
only as explanatory diagnostics. An additive likelihood offset alone cannot
reject a normalized posterior. No probe comparison establishes posterior
fidelity, convergence, posterior discrepancy, finite-posterior existence,
tail truncation accuracy or default readiness.

Measure compile-inclusive and subsequent batch costs and forecast the existing
nested trapezoid design: 64/128/256 intervals across four prior SDs and 384
intervals across six prior SDs. Use the maximum observed steady batch cost,
a factor 1.25 reserve and the measured compilation cost. This forecast
nominates an affordable reference design; it does not authorize treating a
coarse grid as a posterior authority. Dense-reference refinement, empirical
allowances and posterior-function comparisons must be specified before that
stage runs.

Heuristic adversaries are the exact Kalman value, independent joint-Gaussian
reference already checked for R, and constant log-density shift invariance.
Evaluate and report central and outer-prior probes separately. Numerical
validity and same-target wiring are vetoes; pointwise force/value differences
and speed are explanatory. A posterior-accuracy ranking is not available from
these probes.

## Assumptions, resources and audit

Batch size four preserves the checked consumer execution scope and avoids
silently transferring DCTR controls to a different lane. FP32/TF32 finite
values and FP64 reference/coordinates are the same as B; precision limitations
remain explicit. The reference-centered probes are a convenience diagnostic;
their inability to establish global posterior support is a stated limitation.
Fixed deterministic probes introduce no new stochastic seed or calibration
selection.

The finite-target pass has a 6,000 CPU-worker / 7,500 GPU-wall ceiling inside
the existing global 54,000 / 226,800 caps. This includes later finite-reference
work only after its design and forecast are checked. Preflight has 600 CPU /
900 wall seconds and a 180 CPU / 180 wall review allowance. At most three
attempts, including localized harness retries, fit within those caps. Versioned
roots are surrogate-hmc-closure-20260922-ar2-finite-NN. Never overwrite earlier
evidence. Review and tiny tests intentionally hide GPUs; the numerical worker
uses trusted GPU access, memory growth and XLA.

Entry: scripts/run_surrogate_hmc_program.py --stage closure
--closure-stages ar2-finite-preflight --output
docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-finite-01

Skeptical audit before implementation: pass for this diagnostic action. The
baseline and finite target are distinct and explicit; no proxy metric promotes
posterior correctness; invalid tails cannot be relabeled as zero density.
The earliest failure could be canonical evaluation outside the posterior bulk,
which the two conditional groups expose before a long quadrature run.
The pre-mortem is that every probe could pass while quadrature later fails
between probes or in unresolved tails. Consequently only wiring, numerical
domain screening and cost may be concluded. Preserve points, values, status
flags, timings, source/data/consumer identities and the forecast in the report
and tensors; record the follow-up decision in the result note.

## Localized harness repair after preflight-01

The first attempt passed 29 tests and reached actual canonical GPU evaluation,
then failed while assembling the report. It requested numerical_failure from
the raw evaluator, but the public HMC endpoint derives that field from
value_valid and outside_declared_support. The repair uses exactly that public
derivation and preserves the separate force-validity diagnostic. A regression
now supplies the actual raw-status shape without the adapter-derived field.
No numerical control, target, point, margin or budget changes. Preserve the
failed attempt (64.635559 CPU / 28.787410 GPU seconds) and retry in finite-02.


## Existing-master localization repair

Finite-preflight-02 found seven typed RESET_CORRECTION failures in outer-prior probes and a dense-reference forecast above the pass cap. The next bounded repair is specified in docs/plans/surrogate-hmc-ar2-finite-localization-2026-09-23.md: replay those rows plus three healthy controls in FP32 and an independent FP64 diagnostic with exact design equality checks. This stage localizes arithmetic sensitivity versus correction failure; it does not authorize quadrature or reinterpret failures as support.
