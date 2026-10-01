# Fail closed on numerically unresolved iAPF stopping

The preserved GPU failure04868/04869 changes the stopping decision at a
threshold adjacent to the original CV. Its first difference is a one-unit FP64
exponential rounding error. The engineering repair should report that decision
as unresolved; it must not choose a different fit/final action, change tau or
implement another transcendental library. The active adapter stays untouched
until the complete adaptive execution and validation call chains qualify.

Evaluate an additional checked owner using the shared native controller at
f7d03a944. Keep the raw action/CV as diagnostics. At iterations where stopping
is eligible (l>k), construct an outward-rounded interval for CV and reject when
tau lies within that interval. The checked action is then a distinct numerical-
resolution error, with finite endpoints and the original tau retained. The
caller must propagate an explicit DiagnosticFailure; no Python fallback is
allowed. Resolved fit/final/capacity decisions must remain identical.

Accuracy provenance: installed TensorFlow build_info records CUDA12.4. NVIDIA's
CUDA12.4.1 Programming Guide, Mathematical Functions, double-precision table,
lists exp(x) maximum error1 ULP over the full range:
https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#mathematical-functions-appendix
The retrieved guide SHA256 is
0a397f97d44def2f94038a1510c0e6b445b03b7fd52f7fe5e2113e04d2835fe0.
This is a declared primitive-accuracy assumption, not proof of every XLA
transformation or a universal vendor guarantee. Verify the actual compiled
intervals independently; any missed original/high-precision value is a veto.

Start with two outward adjacent FP64 values around each computed exponential
(covering the documented one-ULP error plus one rounding unit); exp(0)=1 is
exact. Propagate interval addition/subtraction/square/square-root/division with
outward rounding at each operation. Use native sequential sums and the already
reviewed division boundaries. A square interval crossing zero has lower bound
zero. The maximum shifted log is zero, so the normalized mean is strictly
positive. Equal-log windows have exactly CV0 and need no widened interval.
No values/weights in the original algorithm are clipped or replaced: these are
separate rejection diagnostics. This is a Class B fail-closed guard under
AGENTS.md, not a new scientific stopping threshold or confidence interval.

Freeze the twelve preceding cases and add24 deterministic histories (k1,2,3,
max_history8) with changed operands, shifts and skewed likelihoods. Require
interval containment of the archived Python CV and an independent80-digit
Decimal calculation, finite ordered intervals containing the candidate CV,
unchanged resolved actions/counts/window flags, exact replay and one trace per
declared signature. All three adversarial adjacent-threshold cases must reject
explicitly on CPU and GPU; ordinary histories must not fire the guard. Also
check invalid histories, no stopping veto at l=k and exact equal-log CV0.
Inspect enclosing HLO and prohibit host callbacks. Do not change these cases
or gates after observing their results.

Allocate6 workers/900 CPU and900 GPU seconds under the existing total: CPU,
GPU, source-bound readback/policy, up to three localized implementation retries.
Use `iapf_resolution_*` groups,300-second timeouts, the stable runner/environment,
new numbered artifacts and one numerical worker at a time. CPU is reference;
GPU uses trusted access and memory growth. A containment miss, changed resolved
decision, nonfinite diagnostic, broken provenance or exhausted budget stops for
repair. No fitting/training/HMC, package/system/cache change, external/model
mutation, subagent or main merge. This guard alone cannot close adaptive iAPF,
costs, ledger validation or public endpoint qualification.

Skeptical review: a fixed absolute tolerance would hide scale/window effects
and merely rename the old mismatch. Outward propagation with independent
high-precision checks tests the arithmetic question directly. The declared
primitive error assumption and compiler sensitivity remain explicit limits.
Intervals may conservatively reject a mathematically decidable case; they
must never silently switch the algorithmic action. This guarded-error behavior
is a deliberate acceptance restriction, preserving the failed exact-parity
evidence. The owner guard directive authorizes its bounded evaluation without
changing the campaign's scientific method, controls or budget.

04871 stops during tracing before numerical evaluation: a Python zero in the
first argument of tf.maximum inferred float32. Use an explicit FP64 zero for
the interval arithmetic. This localized dtype repair changes no gate or formula.

04872 exposes the same literal-inference issue in tf.where before numerical
evaluation. Replace all first-branch interval zero literals with the same FP64
constant; dtype review also covers the equal-window return. No gate changes.

04873 passes all36 CPU resolved/ambiguous histories and high-precision interval
containment. Review found the checked wrapper lacked its own invalid-input
coverage despite raw-controller coverage. Add the seven declared invalid cases
and rerun CPU before GPU/readback; this fits the six-worker allocation.
