# R4a paper-target APF value port and legacy endpoint audit

Question: does a TensorFlow value kernel implement the SAME initial law,
Gaussian-plus-floor proposals, ESS resampling, accumulated weights and terminal
likelihood as the checked R reference? Does the legacy execute_iapf consumer
actually meet this paper-specific target?

Source: publisher paper Section 5.2, Eqs. (5)-(6), Algorithm 5; local reading
text lines 981-1034; independent reference_iapf_paper.R:40-166,416-481.
The Section 5.2 model has initial N(0,I), Q=C=R=I and
A[i,j]=.42^(abs(i-j)+1). No extra transition before the first observation.
Algorithm 5 resamples iff ESS <= kappa*N and carries weights otherwise.

Inspection found the legacy execute_iapf -> make_fitted_twist_kernel ->
parameterized_model chain uses an extra X0 transition, unconditional resampling
and a different six-parameter score-study model. That can define a different
valid filtering problem, but is wrong relative to a Section 5.2 replication
claim. It also uses a bounded profiled fitter with a convergence veto, not the
new finite joint R/BFGS recipe. Do not silently change legacy score semantics.

Implementation: a separate candidate, value-only paper-target TF kernel accepts
explicit model matrices and random inputs, rather than reusing that incompatible
model constructor. The master diagnostic directly calls this shared candidate.
No learned fitting, scoring, new default or HMC claim. The existing consumer
wiring is checked at its actual endpoint and its extra transition and compulsory
resampling are checked with deterministic inputs.

Evidence contract: match the R reference on all particle clouds, log weights,
ancestors, ESS, resampling decisions, untwisted prefixes and terminal values.
Finite floating outputs must agree to 1e-8*(1+abs(reference)); ancestors and
resampling flags must agree exactly. Exact-twist terminal likelihood must match
Kalman to the same tolerance. A mismatch vetoes parity/promotion. It triggers
local implementation repair; it does not reject the method. Invalid source
fixtures or missing required outputs veto interpretation.

Fixtures: d=2,N=32,T=6, constant/observation-floor/exact twists, kappa=0,.5,1;
plus frozen dataset1/rep1/A_raw fit from the completed pilot, using its actual
particle count and d=5,T=100. These are fixed paired inputs, not new tuning data.
R's proposal draw and categorical sampling functions receive shared Gaussian
and uniform arrays; the original APF recursion remains unchanged.
Save all inputs and outputs. Do not rank methods using this parity exercise.

Default/assumption audit:
- FP64 follows the R numerical reference. It is a parity precision, not a
  performance default; FP32/TF32 promotion remains untested.
- N32 is a mechanics fixture, not a paper-scale claim. Include the saved pilot
  scope to catch horizon/particle specializations and report both separately.
- Tolerance is a numerical agreement bound, not statistical accuracy.
- Fixed random inputs allow deterministic parity; branch changes fail exact
  genealogy parity and must be investigated, not averaged away.
- Explicit stable TensorSpecs; one trace per (d,o,N,T,dtype,backend), expected
  two signatures here. Repeat once after first call and record trace counts.
- Native tf.while_loop over time, vectorized particle linear algebra; no pfor,
  NumPy numerical runtime, Python callbacks, I/O or strings in the kernel.
- GPU memory growth is mandatory and verified before device initialization.
  Escalated nvidia-smi then TF physical-device/memory-policy probe precede work.
- CPU and GPU graph/no-JIT are explicit diagnostic exceptions before XLA.
  Run each backend in a fresh process. XLA defaults on in the kernel factory.

Resource limits within existing parent and 12-attempt tranche: fixture export
60 CPU seconds; CPU parity 120 seconds; GPU graph 180 seconds; GPU XLA 360
seconds, at most two attempts per phase. Count failed attempts and compilation.
The GPU ceilings are part of the existing R4 allocation, not extra hours.

Artifacts: master attempt directories, source snapshots, per-case CSV fixtures,
structured comparisons, legacy-call-chain report, wall/compile/repeat times,
host peak RSS, device allocator peak/current, versions, CPU/GPU/TF32/JIT,
verified memory policy, source/fixture checksums, and explicit nonclaims.

Skeptical pre-run review: PASS for this limited value-port claim. It fixes the
wrong target/resampling baseline identified by code and paper inspection.
The complete finite BFGS fitter, adaptive fitting controller, gradients and
MDPF consumers remain unported; no value-only result can close those gaps.
No scientific superiority is inferred. This is a focused prerequisite to R4,
not completion of R4 or either replication. Review is self-review.

Attempt06 repair: R superassignment found the locked base sample.int binding
before any filtering. Inject the deterministic categorical sampler with an
explicit global shadow binding, leaving the base library and APF recursion
unchanged. Ensure its indices are integers. Charged 0.164636224 CPU seconds.
The dispatcher also now propagates a failed child's status to its CLI exit code;
a focused regression checks that external supervisors cannot mistake it for
success. Frozen parity criteria, total budget and phase attempts are unchanged.

Attempt08 numerical localization and reviewed repair: the first eight cases
passed. Exact twist/kappa=1 failed clouds, ancestry, retained weights and
untwisted prefixes while terminal Kalman agreement passed. The frozen R
fixture resampled only twice, although mathematically ESS=N. Cauchy-Schwarz
(sum w)^2 <= N sum w^2 proves ESS<=N, so Algorithm 5's kappa=1 endpoint must
always resample after the first observation. Floating reductions can produce
ESS>N by roundoff and change genealogy. Both R and TF now explicitly implement
that endpoint. No near-boundary tolerance is introduced for kappa in (0,1).
The exporter asserts kappa=1 gives T-1 resamplings and kappa=0 gives zero.
This correction changes floating endpoint trajectories, preserves the exact
algorithm, and leaves healthy interior decisions untouched. Prior fixtures and
attempt08 remain immutable evidence; fixture phase v2 issues a fresh paired
reference. R2 pilot uses kappa=.5 and is unaffected. A fresh fixture, CPU retry,
GPU graph and GPU XLA remain within the twelve-attempt tranche and existing
time ceilings. This is a localized numerical repair, not a relaxed parity gate.

Attempt10: all ten paired paper-target cases pass after the exact endpoint
repair. A separate legacy consumer diagnostic fails because its non-XLA graph
returns three empty tensors instead of scalar/score/clouds. A tiny CPU debug
probe reproduces this with zero and nonzero observation coefficient; the same
callable's eager body returns the expected scalar, score and (2,4,1) clouds.
This is an additional legacy execution defect, not evidence against the new
paper-value kernel or a claim that the legacy GPU/XLA lane was checked.
The audit now records the invalid graph shapes as a legacy-lane veto and checks
its model/initial/resampling equations using that exact callable body under an
explicit eager diagnostic exception. It never substitutes that body at runtime
or marks the graph valid. Endpoint-to-factory wiring remains an executable
sentinel check. Legacy graph localization remains an explicit later repair.

Retry review: allow a third CPU parity attempt and up to sixteen attempts in
the tranche (previous cap twelve). This absorbs localized harness/numerical
repairs without changing scientific criteria, data, method, hardware, CPU/GPU
total limits, or phase time ceilings. Preserve the prior budget configuration.
Preflight runtime provenance is now saved before comparisons so failure cannot
discard hardware and dtype records. No new compute authorization is required
under the existing campaign-repair policy.
