# Five remaining library pfor sites repaired

Both Contract E streaming reset-JVP direction maps now use one shared
TensorFlow while_loop over the parameter-direction axis. The B,N,d batch/cloud
axes and existing analytical reset authorities are preserved. The forward
scalar, source-moment/weight terms, residual-design/ridge dependence, flow and
transport equations are unchanged. The public VJP's fixed dictionary selection
is written explicitly so full-file iteration coverage requires no new allowance.

FixedTransportValueScoreAdapter's two explicit scalar-transport fallbacks use
TensorFlow map_fn with output signatures and parallel_iterations=1. Its default
require_batch_native=True still requires batch methods and rejects scalar-only
transports. An explicitly requested scalar fallback remains ineligible for
batch-native training and does not acquire XLA/HMC admission. The independent
SIR reference grid scout uses persistent GradientTape with
experimental_use_pfor=False and releases the tape after its Jacobian.
No runtime NumPy, autodiff filtering score or Python numerical loop was added.

Fresh B2,N8,d2,P3,float64 Contract E inputs use seed81101 and exact K=N=8 chunks.
No pre-August21 LEDH fixture/result was reused and no old pfor route executed.
CPU04713 and GPU04714 pass total analytical derivative checks against the
unchanged forward scalar's five-point differences at two step sizes, fused
versus separated execution, and forward/reverse derivative pairing. Complete
agreement gates remain1e-9; finite-difference gates remain2e-6. Maximum observed
CPU finite-difference error is1.4872e-12. Gap condition proxies are about2.07--2.17,
all factors finite/positive, and separate direct source/weight adjoint terms are
nonzero. A transported-cloud-only partial derivative would fail these checks.
These are primitive execution/derivative facts, not canonical LEDH admission.

CPU04716 passes28 scalar-fallback/default/admission/adapter tests; GPU04717
passes both rank2/rank3 forward and rank2 logdet cases under fixed-signature XLA.
The fresh scalar-only affine fixture has exact determinant3.744 and preserves
closed-form outputs, dynamic coordinates, one trace and no pfor/host callback.
The default batch-method requirement, batch-native value/score rejection for
scalar-only transports and false XLA/HMC capability are retained. CPU04718
checks two reference-scout Jacobians against finite differences and produces
three finite grids. It is an independent CPU reference only.

Failed attempts are preserved:04711 supplied vector terminal epsilon to a
scalar-only kernel;04712 supplied vector scaling to a scalar tangent-schedule
operation;04715 reused an unsuitable fixture whose inherited batch logdet used
unsupported XLA LogMatrixDeterminant and did not reach the intended fallback.
The latter has26 passes/2 failures. Three reviewed fixture repairs corrected
those concrete call-contract defects without runtime/tolerance changes. The
12-worker and3600 CPU/2400 GPU caps were never increased. No passing input was
selected in response to numerical failure.

Current-source readback/policy04719 passes162 checks. Scope coverage grows279
to282 sources without changing1436 existing exact allowances. Contract E is
covered in full. The scalar transport module guards all pfor/NumPy/default-XLA
operations, with Python-iteration coverage limited to the two touched methods;
other host/configuration iteration remains outside this unit's claim. The named
independent SIR reference retains its permitted NumPy/host-loop role while its
pfor calls are guarded. No admitted runtime file imports that reference in the
BayesFilter source scan. New tests pass Ruff/whitespace checks; the touched
legacy modules retain their3/2/3 preexisting Ruff findings with no new rule/message.

The broader2009-file AST discovery is preserved in
filter_gradient_remaining_pfor_discovery_20260929.json. Within bayesfilter,
the only remaining name-based hit is ValidationTarget.log_density calling its
own explicitly implemented log-coordinate-Jacobian formula, not a TensorFlow
derivative API; readback verifies that method and source hash. Outside that
library,28 sites across17 owned runners/benchmark scripts still require individual
call-chain, approval, repair or retirement review. Neither diagnostic naming nor
an old benchmark date establishes pfor approval. Whole F14 and master stay open.

|Decision|Primary criterion|Veto status|Main uncertainty|Next action|Not concluded|
|---|---|---|---|---|---|
|Five library/reference pfor sites repaired|Fresh CPU/GPU derivative/default-gate evidence and source readback pass|No pfor/host callbacks in tested owners; invalid capabilities reject|Public consumer/default/signature and cost evidence is narrower than whole repository|Review28 runner/benchmark sites and registered score consumers|Whole F14, canonical method or HMC admission|
|Reference scout remains diagnostic|Non-pfor Jacobian agrees with finite differences|Reference is not imported by runtime source|Grid accuracy and posterior status are separate|Keep explicit reference role|Default filtering backend|

|Inference status|Finding|
|---|---|
|Hard veto screen|Fresh tested numerical/derivative/default-gate cases pass|
|Statistically supported ranking|None attempted|
|Descriptive-only differences|Worker times are accounting; no pfor before/after speed ranking|
|Default readiness|Still requires registered consumer and full master gates|
|Next evidence needed|Runner/benchmark pfor dispositions, stable public score owners and applicable memory/cost evidence|

Skeptical terminal review: this verifies the fresh finite scalar and analytical
tangent composition, not every LEDH scientific route. Source-wide discovery
finds debt outside the library, preventing a false whole-repository conclusion.
Scalar fallbacks retain their explicit non-training boundary. The largest
remaining evidence limitations are broader consumer coverage and lack of a
permitted pfor cost comparator. No training/HMC/source-faithfulness claims,
canonical NeuTra architecture edits, tolerance relaxation or main merge occurred.
Primary-agent review only; no independent review asserted.

The completed unit used9 sequential workers,124.740567 CPU and68.071852 GPU
process-seconds. Charges through04719 are109340.306604 CPU/97102.946939 GPU
seconds, leaving25.627693 CPU/25.026959 GPU hours under the unchanged global
caps. No worker is active. Raw04711–04719 evidence and final source snapshots
are archived in artifacts/filter-gradient-repair-20260917/remaining-pfor-04719-evidence.tar.gz.
All52 members reopened and verified;3,929,140 bytes; SHA256
724ec4dd2e3d39b6fac0d57d2182196a624f4e6d93139e4a1a62adbaaa1a54a2.
The adjacent verification JSON records every member hash. The result snapshot
inside the archive precedes this receipt paragraph. Next reviewed plan:
filter_gradient_pfor_runner_closure_20260929.md.
