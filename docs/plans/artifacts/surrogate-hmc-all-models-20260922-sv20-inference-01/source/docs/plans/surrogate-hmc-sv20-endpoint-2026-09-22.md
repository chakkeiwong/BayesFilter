# SV20 claim endpoint and complete inference cost

The goal remains testing endpoint-corrected surrogate HMC with the full
canonical analytical DCTR proposal field. The next ready family is actual
SV20. Its existing likelihood, posterior reference, canonical consumer and
scoped controls are qualified, but it has no retained inference result. This
stage checks the actual claim-data endpoint and measures a complete B/R run.
It does not test a KSC mixture approximation or reclassify preparation as
posterior inference.

## Evidence contract and research intent

Keep the existing attempt1 claim dataset, raw-return observation law,
stationary initial distribution, unit process variance, independent
Uniform(.1,.9) priors on gamma and beta, probit coordinates, N=K=1024 and
selected protection multiplier0.1. Use the reference, posterior, consumer and
control roots dated20260920. Claim observations and the claim particle banks
remain distinct from calibration and validation. No data regeneration or
control selection is allowed here.

The B endpoint uses the existing FP64 1024-interval quadrature on[-16,16].
The comparison reference R uses the qualified full-prior posterior quadrature,
including its1280-interval[-20,20] refinement. Both are numerical
approximations to the same actual-SV law. No exact model-likelihood claim is
made. The DCTR force is an imperfect proposal field, not its derivative.
Finite-DCTR C/D, D/R and C/R remain separate, unimplemented obligations.

The primary criterion is executable qualification: exact claim binding,
reference/consumer value and score parity, explicit domain-mass validity,
deterministic replay and row permutation, independent endpoint/force bindings,
direct canonical force call-chain parity, reversible proposal and healthy
complete transitions. Compare endpoint values and scores against the finer
reference at the nine physical parameter pairs in `{.11,.5,.89}²`. These
cover the center and each near-boundary combination without changing the
existing full-prior reference. Retain
the earlier absolute log-likelihood tolerance1e-6, relative score tolerance
1e-5 and predicted domain-loss limit1e-9. A finite but inaccurate endpoint,
wrong law, missing prior/Jacobian, force fork, numerical failure, source/data
drift or missing diagnostic vetoes qualification. Repairable implementation
failures do not reject surrogate HMC.

Emit the already-computed domain-mass diagnostic in the candidate endpoint.
A domain loss above1e-9 marks numerical invalidity, never true prior support
exclusion. This fail-closed guard changes no accepted density or derivative.
A no-fire comparison against the qualified endpoint is required. Do not
renormalize lost state mass, floor a likelihood, alter DCTR or relax limits.

Force/reference gradient differences, acceptance, costs and allocator use are
explanatory. A cost forecast exceeding available allocation vetoes the next
launch, not this method. This stage cannot establish posterior agreement,
convergence, superiority, broad SV validity or default readiness.

## Default audit and prospective cost schedule

| Choice | Provenance and reason | Failure mode and earliest check | Status |
|---|---|---|---|
| Actual SV20 law, prior, chart and once-rounded claim returns | Existing source-defined target and qualified posterior | Accidentally using transformed KSC law; data/law and reference parity | Fixed scope |
| FP64 1024-interval endpoint | Existing qualified actual-SV consumer; full-prior refinement changed log likelihood by less than1.5e-14 | Tail/domain loss or lost normalizers; finer reference and explicit mass guard | Numerical reference hypothesis |
| Canonical FP32 DCTR and multiplier0.1 | Existing scope-specific independent control selection | Wrong callback/bank or failed full feature path; callable parity and status traces | Frozen proposal candidate |
| One leapfrog step for the prospective first B attempt | Existing SV20 timing makes a longer search costly; L1 is a valid restricted endpoint-corrected proposal | Poor movement or mixing; fresh public-tuner verification and retained uncertainty screens | Cost-constrained experiment, not an optimal HMC setting |
| Mass windows32/64; adaptation128 and nomination verification64; fresh warmup250/verification1000; retained1000/2000 | Same public interface and evidence procedure as earlier scopes, used only as an initial hypothesis | Inadequate tuning/precision; explicit failure instead of promotion | Bounded diagnostic inference schedule |
| GPU1, TF32 force, FP64 reference/integrator, XLA and memory growth | Repository execution policy and existing qualifiers | Device mismatch, numerical drift or allocation failure; manifest and parity | Required execution configuration |

Measure five complete four-chain L1 transitions, retaining compilation cost
separately and using the maximum of four steady calls. Measure the actual
32-row model-predictive functional batch used by future retained diagnostics,
with one compilation call and four steady calls over repeated qualification
positions. Keep peak TensorFlow allocation below 4 GiB, matching the earlier
consumer qualification.
Forecast1.5 times the sum of3586 transitions,256 predictive batches,240 seconds
for compilation and90 for analysis. The transition count includes96 mass
adaptation steps,192 candidate steps,48 additional checks,1250 fresh warmup
and verification steps, and2000 retained steps. Predictive evaluations are
cached by retained chunk: eight chunks of1000 draws need32 padded batches
each, rather than recomputing earlier functionals at each look. This is
prospective: no HMC
tuning or retained chains run in the present stage. Any later launch needs the
completed measured forecast, a recorded allocation within the unchanged
global caps, and a frozen inference plan.

Heuristic adversaries remain prior-center conditional prediction, prior
predictive integration and a Gaussian/Kalman moment approximation, evaluated
separately in low/high absolute-return conditions. This stage does not run
those comparisons; their machine-readable verdict remains NOT_RUN and they
block usefulness, efficiency and default promotion.

## Execution, budget and skeptical audit

Use the tftwogpu Python environment, escalated GPU1 UUID
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, memory growth, two CPU threads and
the new `scripts/qualify_surrogate_hmc_sv_endpoint.py` command. Run only after
the active M10 worker closes. Use a fresh
`docs/plans/artifacts/surrogate-hmc-all-models-20260922-sv20-endpoint-01`
root, preserving report, manifest, complete probes, source snapshot, worker
log and accounting. Reserve450 CPU-worker seconds/600 wall seconds, plus40
CPU seconds for focused checks, within the existing SV-family allocation.
The parent must bound resources and charge failures. Stop on a failed
qualification, corrupted input, budget exhaustion or hardware failure. Permit
one localized infrastructure retry only within this same reservation.

Skeptical audit: the likelihood and posterior references use the raw-return
law and existing priors, and the canonical field remains analytical. The
current consumer discards a computed mass-loss diagnostic; the new guard must
retain it without changing accepted results. The earlier timing was an
optimistic incomplete lower bound, so it cannot authorize a run. Restricted L1
may mix poorly and two looks may be underpowered; those outcomes must remain
inconclusive, with no equivalence or ranking inferred from numerical health.
These checks make the qualification informative before a larger allocation.

Pre-mortem: a successful command could still compare a calibration endpoint
with a claim posterior, lose the raw-return normalizer, or hide truncation by
renormalizing mass. Explicit binding, source checks and finer-reference parity
address those risks. No new method, tuning result or statistical look is
claimed by this preparation.
