# Remaining pfor execution migration

Continue after the optional-batch disposition. Engineering question: can the
remaining pfor/implicit-pfor sites preserve their declared numerical contracts
using TensorFlow loops or explicitly non-pfor Jacobians, without weakening
batch-native training/default gates? This is execution repair, not a canonical
LEDH rebuild, new Zhao-Cui source route, transport training or HMC campaign.

1. Contract E streaming JVP: both _contract_e_streaming_jvp_core and
_contract_e_streaming_forward_jvp_core map analytical reset tangents over the
last parameter-direction axis using pfor. Replace only that map with one shared
native TensorFlow loop; preserve B,N,d axes and unchanged single-direction
_contract_e_chol_cloud_jvp_core / _jvp_from_forward_core. No sample-row mapping
is introduced. Preserve direct source moment/weight, transport, design/ridge
and epsilon0 derivative contributions. The forward-only scalar and analytical
formula remain untouched. Callers include canonical LGSSM, latent SIR and the
Zhao-Cui moment teacher; no source-faithfulness or caller admission claim follows.

2. FixedTransportValueScoreAdapter: default require_batch_native=True requires
batch transport methods and must retain that gate. The explicit false setting
has two unapproved scalar-transform pfor fallbacks. Replace them with a native
TensorFlow map/loop with declared output shape; it remains a diagnostic scalar
fallback, not an eligible training route. Preserve rank/sample-axis behavior,
constant/frozen transport configuration and dynamic input operands. Existing
batch-native forward/logdet routes are unchanged. Do not alter canonical NeuTra
IAF architecture, kernels, masks, training, or automatic differentiation roles.

3. sir_latent_preclip_reference_tf.prepare_reduced_dense_grids: explicitly
reference-only scout retains its diagnostic NumPy and host loop role. Request
experimental_use_pfor=False from its GradientTape Jacobian, with persistent tape
where eager non-pfor requires it, then release the tape. This does not promote
the scout to a runtime candidate or analytical parameter-score authority.

Controls/baselines: freeze current source before edits. Do not execute the old
pfor implementations without applicable approval. For Contract E use fresh
seeded B2,N8,d2,P3,float64 primitive inputs with exact K=N chunking; avoid all
pre-August21 LEDH result/fixture artifacts and wrong historical chunk policies.
Use the unchanged forward scalar, independent five-point finite differences,
analytical one-direction calls, JVP/VJP pairing, and separate/fused forward/JVP
agreement. Check source-particle-only, weight/log-weight-only, and all-controls
directions so a transported-cloud-only partial derivative cannot pass. Keep
healthy1e-9 internal agreement and2e-6 finite-difference gates. Record covariance
validity/conditioning and reject/diagnose ill-conditioned cases; do not increase
tolerances. Use explicit fixed-signature XLA owners, changed operands, one trace,
no pfor/host callbacks and CPU/GPU placement/growth provenance. Independent
reference autodiff is diagnostic only; no runtime autodiff score is introduced.

For the transport fallback use a fresh scalar-only affine fixture with exact
closed-form forward/logdet/score, dynamic batched coordinates, and the unchanged
require_batch_native rejection gate. Verify rank2 logdet and higher-rank sample
forward handling against prior declared behavior; unexpected ambiguity triggers
localization, not a silent semantic change. Run existing batch-native/admission
and canonical transport tests appropriate to this touched call path, without
training. The reference scout gets a small fresh fixture and comparison with
an analytical/finite-difference transition Jacobian, no historical LEDH results.

Audit/guard scope: add exact pfor and implicit-pfor guards over these files
without broad NumPy/host-loop exemptions. The scout is explicitly independent
reference and is never imported by runtime candidates. Classify legitimate
configuration/reporting structural iteration; do not disguise a numerical
loop as metadata. Full-file default-XLA/source-signature migration is separate
where this bounded unit does not establish it. F14 closes only after all
original audit sites plus newly discovered sites have verified dispositions.

Allocation: at most12 sequential workers,3600 CPU/2400 GPU seconds within the
unchanged56 CPU/52 GPU process-hour global caps. Up to2 localized harness
retries inside that allowance. Stable registered runner prefix;300-second
workers initially. No broad cache/system/environment change, no subagents,
training, HMC, external publication, live MacroFinance edit, new tolerance,
scientific/default admission or main merge. GPU remains default; CPU explicit
reference only. Recheck non-display availability and growth before GPU runs;
shared numerical evidence is not timing evidence. Preserve exact commands,
source/environment/seeds and unique raw artifacts under the campaign root.

Pre-mortem/review: pfor removal could lose leading dimensions or omit the direct
moment derivative; separate source/weight directions and exact output-shape
checks are vetoes. Finite differences alone could hide branch instability;
check two step sizes and report condition/validity. A batch-native flag cannot
admit a scalar fallback; existing gates must remain effective. Diagnostic-only
NumPy in the scout does not authorize runtime imports. Skip old fixture-driven
LEDH phase4 tests as new evidence when their data/results predate invalidation;
build fresh primitive fixtures. Any missing source/ownership/signature or
numerical drift stops that unit for localization. Primary-agent review only.

Attempt04711 stopped before numerical execution: the fresh fixture supplied
vector terminal epsilon, but the shared manual finite transport contract
requires scalar epsilon. Keep epsilon=1 as a scalar; epsilon0/ridge and their
declared per-batch tangent shapes remain as specified by the shared kernel.
This is localized fixture repair1/2, preserving controls/distribution/method and
tolerances; no runtime repair or relaxed gate follows from the fixture error.

Attempt04712 stopped while tracing the unchanged transport annealing schedule:
the fixture also supplied vector scaling, but the shared direction update
requires a scalar schedule factor. Localized fixture repair2/2 uses scaling=.8
as a scalar, preserving epsilon0=[2,2] and its B-by-P tangent. No runtime source
or tolerance changes. The two fixture issues consume the declared harness
retry allowance; any further failure must be localized before another launch.

Transport attempt04715:26 existing adapter checks pass. Both new fallback tests
fail because the reused NoBatchTransport fixture still exposes a batch logdet
method and uses unsupported XLA LogMatrixDeterminant. This is a concrete
fixture/call-chain flaw: it cannot exercise the intended missing-batch logdet
fallback. Replace that reuse with the fresh scalar-only affine fixture already
required by this plan; assert both batch methods are absent, use its exact
fixed determinant3.744 and preserve the same matrix/shift/data/gates. No
runtime change is needed. This mandatory prerequisite review permits a third
localized fixture repair, increasing only that local retry allowance2 to3;
the12-worker/3600 CPU/2400 GPU ceilings and global campaign budget are unchanged.
All three fixture failures are retained. Stop/review any further harness failure
before another numerical attempt; do not select a passing input or relax gates.
