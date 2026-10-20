# Replication continuation readiness, 5 October 2026

Completed continuation, 5 October 2026: attempts149–151 finished the [fitting/forward comparison](controlled-smc-p0-fitting-forward-result-20261005.md) and [conditional solver reference](controlled-smc-p0-solver-semantics-result-20261005.md). All 32 TF fits/forwards were valid but failed the strict matched-draw screens; the Octave-linked independent solver reproduced all 128 saved coefficient vectors bit for bit. Full TF fitting and P1 qualification remain open. No numerical job is running; E0 remains deferred.

Question: which remaining replication steps can proceed, and which still need
a specified experiment or a demonstrated repair? This note updates the
[master](controlled-smc-kdm-replication-master-program-20260928.md). It does
not authorize changing the author's method or launching E0.

## Checked starting point

Attempts145–151 are terminal; no campaign worker or queue remains active.
Both original-source neuroscience matrices contain 2000 completed cases.
The Lorenz source completed 3537 cases before its preserved covariance failure.
The 22 valid fixed-policy forward comparisons pass, with exact ancestry.
Attempt 148's aggregate verdict remains FAILED_POLICY_FORWARD_NOT_CONFIRMED:
its exporter retained the complex policy in MAT form and correctly declined
to turn it into a real-valued fixture. Historical results remain unchanged.

Readiness inspection of attempt 146's original MAT file found all A, b and c
finite, but genuinely complex. Their maximum absolute imaginary parts are
2.60646204154814e-5, 0.001000969464806227 and 1.5626824795441672 respectively.
Calling this policy nonfinite is inaccurate. Stripping its imaginary parts
would create a different input and is forbidden in this comparison.

## Immediate check and skeptical audit

Run a routine CPU-only input-binding smoke at the actual
MultivariateGaussianControlledSMC.forward endpoint using the exact complex
policy tensors, with their original shapes and values. Preserve the MAT hash,
implementation hashes, endpoint signature, exception classification, trace
counts, environment and elapsed time. CUDA is hidden before framework import.
Keep jit_compile=True, but this check must reject inputs before tracing or
kernel execution. Dummy observations/draws have correct shapes and cannot
support a trajectory comparison because no numerical kernel should execute.

Pass means a TypeError explicitly identifies complex128 versus float64,
with zero traces before and after. Any acceptance, unexpected exception,
silent real cast or kernel execution requires investigation. This is an
input-contract smoke, not a numerical campaign or P0/P1 promotion run. It
does not consume a new numbered experiment attempt. Limit: 90 CPU wall
seconds, zero GPU seconds; report the routine-check cost separately.
NumPy/SciPy are used only to inspect and import the saved independent source
fixture; they are not an implementation backend.

Skeptical audit: an ordinary dtype rejection can establish the real-valued
consumer's input boundary, but cannot demonstrate that TensorFlow fitting
recreates the source's failed trajectory. Do not retrospectively change
attempt 148's criterion, call the smoke GPU/XLA numerical evidence, or treat
it as full forward/fitting parity. Healthy-case numerical evidence remains
the existing 22 GPU/XLA comparisons. No source numerical safeguard changes.

Output root:
artifacts/controlled-smc-kdm-replication-20260928/readiness-20261005-01/.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/readiness-20261005-01/input_binding_smoke.py

## Ordered continuation

| Step | Ready now? | Required action and completion evidence |
|---|---|---|
| Invalid source policy accounting | Input-boundary check complete | Exact complex input rejected before tracing. Numerical failure-trajectory parity remains separate and unproved. |
| Lorenz fitting translation | Initial comparison and CPU reference complete; TF repair unproved | Attempt149:all32 fits/forwards valid but both solvers fail the strict screens, with passing source-policy controls. Attempt151:all128 source coefficient vectors reproduced bit for bit. Bind TF condition estimation, conditional branches and source rank semantics; all60 fallback ranks were45. Check the actual fitting/forward consumers and freeze independent P1 criteria. Smaller residuals or coefficient screens alone cannot admit a solver; no new truncation/ridge objective. |
| P1 no-worse qualification | Not executable yet | Freeze scale/conditioning-based deterministic tolerances and stochastic non-inferiority margins with separate calibration before untouched comparisons. Count intrinsic source failures as reference outcomes; count additional port failures separately. |
| Neuroscience paper analysis | Data available; current all-four-driver C2 command blocked | Add an explicitly scoped analysis of the two completed neuroscience matrices if needed. Do not relabel partial Lorenz evidence as a complete Figure 6 result. |
| Lorenz Table 1 / multimodal | Separate specifications required | Table 1 can reproduce the generative design, but original unseeded datasets are unavailable. Multimodal has a missing driver and a 100-versus-101-observation ambiguity. Preserve these limitations; do not substitute a quadratic policy for the RBF/NNLS source method. |
| KDM/IWSG | Training chain not ready | Bearings configs exist, but component pretraining, MDPF-before-A-MDPF dependencies, full learning-update parity, independent trials and cost remain. Appendix B's exact numerical initialization and schedule are unrecovered. Do not substitute the nonlinear toy dataset or call a reconstruction exact replication. |

The fitting/forward and conditional-reference phases have now run. For the next TF implementation/qualification run, first add its concrete executable phase,
unique output directory, focused regression checks and bounded attempt
allocation to the dispatcher. Its contract must preserve the source objective,
settings and accepted failure behavior. Existing authorization permits these
localized repairs within the unchanged campaign budget; no new owner approval
is needed merely to update exhausted bookkeeping. Missing scientific criteria
are not waived by that authorization.

Current numbered attempt ceiling is151 and all151 are spent. Remaining
active numbered allocation is151727.535367 CPU seconds (42.15 hours) and
2812.436807 GPU seconds (46.87 minutes); routine CPU checks are recorded separately. The
larger inherited GPU balance is not an active launch allocation. This supports
bounded diagnostics; full independent KDM training has not been costed, so
there is no evidence that the current GPU allocation can complete it.

Stop an affected experiment for wrong inputs/baseline, corrupt artifacts,
missing required diagnostics or exhausted budget. A matched intrinsic author
failure is an accepted baseline outcome, not an instruction to repair the
author. Full replication remains incomplete.

## Executed readiness check

The CPU-only input smoke passed in 2.603926 seconds. The actual endpoint
raised TypeError identifying complex128 versus float64. All preserved policy
values match their MAT arrays exactly; the MAT hash matches attempt 146's
manifest. The consumer and dependency hashes match attempt 148's healthy
comparison snapshot. Trace counts were zero before and after, so no forward
numerical kernel ran. GPU time was zero. The CUDA_NO_DEVICE initialization
message is expected with GPU devices deliberately hidden; it is not a GPU
health diagnosis. Full command output, exception and provenance are preserved
in readiness-20261005-01/command.log, rejection.txt and summary.json.

| Decision | Criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Input boundary checked | Exact complex policy rejected before tracing | No cast, acceptance or unexpected exception | This tests binding, not fitted-trajectory behavior | Retain alongside completed fitting comparison | Complete P0 or P1 |
| Continue bounded replication work | Conditional CPU reference exact on128 systems; TF fitting comparison completed | Strict TF policy/downstream screens failed; no additional invalidity observed | Full fitting repair, downstream non-inferiority and KDM training costs remain unproved | Specify TF condition/branch/rank behavior and P1 criteria | Both replications ready for unattended completion |

Terminal self-review: a rejected complex input is weaker than reproducing
the numerical failure that generated it. The unchanged attempt 148 verdict
and the outstanding fitting/trajectory checks preserve that distinction.
No statistical ranking or default-readiness claim follows from this smoke.
