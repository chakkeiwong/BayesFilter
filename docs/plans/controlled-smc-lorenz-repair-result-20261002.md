# Lorenz repair: accepted bookkeeping fix, unresolved adaptive-policy instability

The default controller now keeps its last accepted policy and particle history
when a refit fails, reports the attempted iteration separately, and counts only
successful refits. Attempt135 passes all20 acceptance checks, including the
complete healthy101-observation,512-particle, four-refit run and an injected
failure. Healthy ancestry agrees exactly. The source-scale numerical repair is
still incomplete: neither the early-stop kernel nor the alternative Bellman
arithmetic has passed the required full coefficient comparisons.

This is the result of the [repair execution plan](controlled-smc-lorenz-repair-execution-20261002.md),
following the [code and mathematics audit](controlled-smc-lorenz-root-cause-result-20261002.md).
The [master](controlled-smc-kdm-replication-master-program-20260928.md) remains
active. Controlled-SMC Figure6, source-M16/Table2 and empirical KDM/IWSG learning
remain incomplete; iAPF remains deferred. These diagnostics add no performance,
posterior, HMC or complete-replication claim.

## What was repaired and checked

In `LorenzRefinement._run`, acceptance is now `active & good`. Both the policy
and particle history retain the preceding accepted values when `good` is false;
the successful-refit counter increments only on acceptance. `valid` remains
false after rejection, so preserving useful diagnostic history cannot turn the
failed run into a successful one. The new `attempted_iterations` and
`rejected_refit` fields distinguish an attempt from a success.

The numerical `_forward`, `_fit`, `_regress` and `_cloud_weights` implementations
used by default retain their pre-repair arithmetic. Attempt135 checks the actual
controller-to-engine call chain. Its injected failed refit preserves A, b, c,
particles, ancestors and logZ exactly, with zero accepted and one attempted
refit. It also reproduces all saved111 four-replica GPU backward coefficients,
Gaussian corrections and projection increments bit for bit.

The optional `first_failure_guards=True` implementation stops before invalid
weights reach normalization or resampling, records the failed time and operation,
and retains the rejected proposal separately. Attempt131 locates the actual121
failure at refit4/time0: its proposed mean is635.0103 and its lookahead RK4
transition becomes nonfinite before any resampling. This is a downstream
consequence of the fitted proposal, not evidence of a broken GPU installation.

The broader guard candidate failed the full healthy coefficient comparison in
132/133. Although ancestry and forward statistics passed, A/b did not meet the
unchanged tolerance after adaptive refitting. It therefore remains an explicitly
experimental path, disabled by default. This is a failed non-harm check for this
implementation of guards, not a reason to abandon failure diagnostics.

A subsequent static review found a separate guard defect: when one replica
failed, placeholder values from skipped operations could falsely label healthy
replicas as failed. The repaired diagnostic keeps only the first executed
failure stage and does not interpret a skipped QR's placeholders as rank
failures. Attempt138 repeats the31 earlier checks and adds eight two-replica
checks: all39 pass. This does not supersede the failed full-scope parity check.

## Why the backward recursion still disagrees

Attempt130 independently replays frozen108/111 row0 using SVD regression and
Gaussian algebra. The first coefficient tolerance failure appears at backward
time12; mean agreement fails at11; Gaussian correction and independent polynomial
probe agreement fail at10. Reproducing the GPU's mixed correction branches
reduces the largest A difference from0.0988950 to0.0585110, but does not close it.
The discrepancy is not explained by that branch difference alone.

There is a mathematical mechanism for amplification. For
q(x)=x' A x+b' x+c, define

\[
L(A,b,c;s,Q)=\log E_{X\sim N(s,Q)}[e^{-q(X)}].
\]

With positive tilted precision, differentiation under this Gaussian integral
gives, for a small coefficient perturbation,

\[
dL=-E_{\mathrm{tilted}}[X'\,dA\,X+db'X+dc]
   =-\operatorname{tr}((C+mm')dA)-m' db-dc.
\]

Here m and C are the tilted Gaussian mean and covariance. This follows directly
by differentiating the integrand and dividing by its integral; the Gaussian
moments are the ones derived from the paper in the preceding root-cause report.
Consequently a large m magnifies a small curvature error through mm', even when
the matrix inversion itself remains finite.

Attempt134 checks this mechanism on actual successive fits. At next-time11,
the mean reaches103.281 and the A difference is only about2.04e-5, but the
log-integral difference reaches0.01591330. The linearization predicts that
difference with maximum error below8e-10 on the saved inputs. Pinning the
next-time11 increment to the GPU's saved increment reduces the downstream
time10 A discrepancy from0.0988950 to0.00030087; the result still fails tolerance.
Pinning another stage can worsen agreement through cancellation, so this is
evidence of coupled sensitivity, not proof of a single faulty operation.

The corrected four-replica comparison in135 reproduces the saved GPU recursion
exactly, while a CPU graph differs by as much as0.541384 in A. Both use the same
frozen inputs. This establishes repeatability of that GPU calculation, not its
independent numerical accuracy. There is no evidence here that retracing,
Python loops over particles, NumPy runtime computation or TF32 caused the
disagreement: the numerical loops use TensorFlow, the tested signatures trace
once, and these comparisons use FP64 with TF32 disabled.

## The algebraic reformulation did not pass

Before projection, write the old incremental weight as
log w_t=log g_t+L_old,t+1+q_old,t. The correction target is

\[
-\log w_t-(L_{new,t+1}-L_{old,t+1})
=-\log g_t-q_{old,t}-L_{new,t+1}.
\]

Because q_old is in the full quadratic regression span and least squares is
linear at full rank, adding the fitted increment to q_old is algebraically
equivalent to fitting -log g_t-L_new,t+1 directly. For this Lorenz observation
model, -log g is itself an exact quadratic. Attempt136 therefore fits only the
future term, adds the analytical observation coefficients, and applies the same
increment projection. Its Gaussian integral uses Cholesky solves and a completed
square. This preserves the intended regression objective while changing its
floating-point evaluation.

Both backends stay finite and trace once, but all five comparison groups fail:
GPU versus CPU maximum A difference is0.260010, and the future-integral
difference is0.0428244. These descriptive reductions are not an admission or a
ranking. The candidate is rejected under the existing comparison criterion.
Its top-level PASS means the diagnostic ran and captured valid values; the
explicit `cross_backend_arithmetic_agreement=false` and
`candidate_admitted=false` carry the scientific verdict.

## What the safeguard investigation does and does not show

Attempt134 also interpolates the complete time0 coefficient increment
(A,b,c) between the old policy and the failed121 proposal, with the downstream
policy held fixed. For each alpha it calculates Gaussian KL(new||old), proposal
geometry, RK4 validity and Bellman-model error on fixed normal probes.

| Initial-update alpha | KL(new||old), approximately | Maximum absolute mean | RK4 on saved probes |
|---|---:|---:|---|
| 0 | 0 | 1.323 | finite |
| 1/1024 | 0.00879 | 1.334 | finite |
| 1/64 | 2.262 | 1.503 | finite |
| 1/4 | 693.2 | 4.520 | finite |
| 0.9 | 249404 | 50.31 | finite |
| 0.99 | 10201749 | 293.3 | nonfinite |
| 1 | 48860065 | 635.0 | nonfinite |

The local model-error RMS grows from about0.00943 at alpha0 to81961 at0.9,
before RK4 overflows. Merely making the transition finite is therefore an
inadequate acceptance condition. These are a calibration curve on one known
failure, not a selected KL limit, a repaired backward sweep, or a fresh
validation experiment. Upstream projection counterfactuals and an admissible
constraint on the complete policy update remain open.

## Execution record and evidence corrections

Each attempt directory below contains `summary.json` when numerical work
executed, `manifest.json`, the full command log and source snapshots under
[the campaign root](artifacts/controlled-smc-kdm-replication-20260928).

| Attempt | Work | Result |
|---|---|---|
| 130 | Independent recursion localization | Captured first disagreements; admission still fails |
| 131 | Guard consumer checks and saved failure replay | 31 checks pass |
| 132 | Full healthy guard comparison | Fails coefficient tolerance; charged and preserved |
| 133 | Full guard capture | Healthy A/b fail; part of recursion comparison invalidated below |
| 134 | Independent sensitivity and initial-update curve | Mechanism measured; no safeguard selected |
| 135 | Default bookkeeping and four-replica recursion | 20 bookkeeping checks pass; GPU exact, CPU parity fails |
| 136 | Equivalent absolute Bellman fit | Finite, cross-backend comparison fails |
| 137 | Mixed-row regression launch | Missing snapshot import; no numerical result |
| 138 | Repaired mixed-row regression | 39 checks pass; broader guard still unadmitted |

The recursion-comparison fields of133 accidentally broadcast a single replica
against four saved replicas. Those fields are invalid evidence. Its separate
one-replica healthy no-fire comparison is valid. Attempt135 replaces the invalid
comparison using matching four-replica shapes, and new comparison helpers
explicitly reject unequal shapes. Earlier independent130/134 used the intended
single row and are unaffected.

Attempt137 failed before TensorFlow import because its isolated snapshot omitted
the first-failure test dependency. The dependency was added and138 ran in a fresh
directory. Failed attempts count against the resource cap. The initial six
attempts were extended to eight localized diagnostics and then one infrastructure
retry under the same30 CPU/300 GPU second limits; no additional resource grant
was assumed. Attempts130–138 charged6.094010 CPU and176.247307 GPU process-seconds,
including compilation and failures. No numerical process remains running.

CPU references intentionally hid the GPUs. GPU runs used trusted access,
verified memory growth on both devices, FP64, XLA, and TF32 off; CPU graph
comparisons are explicit reference exceptions. Synthetic guard fixtures use
stateless seeds20261002/1 and20261002/137. Earlier generic manifest seed wording
omitted these synthetic fixtures, although the exact seeds were in the saved
source. Attempt138 records that distinction. See the terminal review for
source/input hashes and remaining provenance limitations.

## Decisions and next work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep default acceptance repair | 20 checks pass, including full healthy scope | No failed repair check | Other execution scopes | Preserve this localized change | Robust fitted-policy default |
| Keep early-stop kernel experimental | 39 localized guard checks pass | Full healthy A/b parity failed | Compiler arithmetic interaction | Separate failure capture from healthy arithmetic; repeat full no-fire comparison before admission | All guards are useless |
| Keep numerical recursion unadmitted | Independent and CPU agreement fail | Unchanged tolerance exceeded | Accurate reference trajectory and error amplification | Resolve sensitive time12/11 with a high-precision local reference and propagated error estimates | Paper or every quadratic policy is invalid |
| Evaluate joint proposal protection | Initial-update curve exposes excessive movement | No calibrated, validated candidate | Upstream causes and healthy non-harm | Test Gaussian KL and model adequacy across frozen healthy/failing cases, then independent full recursion | A chosen damping coefficient |
| Defer empirical scale-up | Eligibility not reached | Numerical and heuristic requirements remain open | Cost after a valid batch implementation | Validate fresh seeds, conditional BPF/APF controls and total fitting cost before Figure6 | Complete controlled-SMC or KDM replication |

| Inference status | Finding |
|---|---|
| Hard veto screen | Full-recursion disagreement, failed broad-guard non-harm and the saved nonfinite proposal are supported |
| Statistically supported ranking | None; these are deterministic repairs and frozen-case diagnostics |
| Descriptive-only differences | Backend error magnitudes and the initial-update calibration curve |
| Default-readiness | Acceptance bookkeeping is repaired; no numerical safeguard or new fit route is admitted |
| Next evidence needed | Accurate independent recursion, calibrated joint-update non-harm, untouched conditional filter validation |

Terminal skeptical assessment: the accepted bookkeeping change is checked
through the consumer, while the broader early-stop code is not smuggled into
the default. The observed derivative explains local amplification but cannot
establish that it is the only source of every recursive difference. A
high-precision propagated reference could identify an additional arithmetic
defect; healthy/failing upstream counterfactuals could change the attribution
to projections. The weakest evidence remains generalization beyond saved
trajectories. New diagnostics must answer those questions before a long
replication run; another covariance-floor sweep would not do so.
