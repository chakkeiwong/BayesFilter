# C2 frozen-proposal preparation execution gap

Current-caller review after05311 identifies a missed in-scope enclosing
execution defect. FrozenProposalAPFProgram evaluates values and analytical
scores through its compiled owner, but callers build the frozen branch with
Python time recurrences. This is fixed proposal construction, not deferred
adaptive iAPF or KDM. Candidate/diagnostic labeling is not an exception to the
default-XLA policy for algorithmic candidate implementations.

The concrete paths at385a348b9 are:

- bayesfilter/highdim/c2_mixture_ukf_apf_c2_adapter.py:
  compile_c2_per_ancestor_ukf_apf_k1 (time loop194), the fixed-mixture compiler
  and the defensive-mixture compiler. Each feeds computed parent weights and
  covariances into the next UKF/proposal step, so this is numerical recursion.
  The K=1 path constructs and evaluates prefixes at189/236 and applies host
  finite checks. A jit_compile=True default on its inner kernels does not
  compile that enclosing computation.
- bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py:
  _compile_c2_branch (time loop967) recomputes exact prefix weights and feeds
  them into the next resampling/proposal step. Public bootstrap, independent
  Gaussian/Hermite and transformed-Student compilers reach it. The DMIS
  compiler has its own time loop682 and exact-prefix feedback. Existing route
  classification remains extension_or_invention; this campaign changes
  execution only and makes no Zhao-Cui source-faithfulness claim.

Neither module is currently in the326-source execution guard. Existing
F06 core tests/costs qualify the prepared evaluator and cannot close this
preparation debt. Keep F06/F19 open and add the concrete callers to the final
inventory. Preserve completed evaluator measurements; a new preparation
repair does not invalidate unchanged evaluator numerical dependencies.

Repair sequence after the source-frozen core cost matrix:

1. Enumerate all public branch compilers and their actual repository callers;
   separate heterogeneous proposal configuration from recurrent numerical
   work. Inspect the source of shared sampling/UKF and exact-prefix weight
   authorities before choosing integration boundaries. Retain current
   classification, numerical method, theta-independent frozen branch,
   component/categorical choices, seed words, diagnostic fields and errors.
2. Freeze independent complete original branch/manifest/value/analytical-score
   records on tiny valid and invalid fixtures for each reachable supported
   compiler family. Explicitly cover ordinary and enclosing random contexts;
   use the existing compatible TensorFlow random authority where applicable.
   No stream substitution is authorized here. First establish a discriminating
   K=1 full-call witness before a larger family matrix.
3. Move time recurrences and numerical preparation into retained TensorFlow
   owners with explicit signatures and default XLA. Pack time-indexed inputs
   and use tensor control flow. Host configuration may build typed/static
   proposal metadata; host validation/reporting may consume completed results.
   Do not replace the analytical-score authority, recompute a scalar target
   per sample, silently fall back, or alter an invalidity/selection decision.
4. Expand guard coverage to actual compiler dependencies. Verify original
   records at existing1e-10 FP64 bounds, exact ancestor/component/status
   decisions, independent filter/score fixtures, dynamic operands, one trace,
   no callbacks and bounded graph growth. Validate GPU/XLA and labeled CPU
   references. Qualify complete public costs and bounded reuse separately
   from the already measured prepared evaluator.

This is a concrete implementation gap and cannot be closed by the ongoing
core cost cohort, a syntax-only disposition or an unsupported-use label.
The matrix remains source-frozen; no runtime/test/harness edit for this unit
may be installed while it runs. Draft work may be kept under/tmp. Before
numerical execution, register a bounded family-specific evidence/attempt
allocation inside the remaining global budget, with exact commands, frozen
source/data, hardware, unique artifacts and stop conditions. This plan does
not launch an unbounded sweep or expand compute authorization.

Primary-agent skeptical review: the misleading-pass risk is again measuring
a compiled inner evaluator while the actual enclosing preparation remains
Python-driven. Prefix rebuilding and random-context conversions can hide
method/rounding changes, so complete branch records and ordered seed/ancestor
checks precede performance interpretation. Static lists of proposal objects
do not excuse numerical time feedback. If a callback cannot be represented
by the supported compiled protocol, identify that concrete API requirement
and preserve explicit failure rather than substituting another proposal.
No new tuning, posterior/HMC, method-quality or canonical LEDH work is included.

Source enumeration records12 functions with iteration sites and39 direct
named calls in six files, saved in/tmp/filter-repair-c2-preparation-source-inventory.json.
This includes host manifest/fingerprint traversal, so12 is not a defect count.
The concrete benchmark callers are run_c2_mixture_ukf_apf_20260902.py,
run_c2_frozen_tt_proposal_apf_20260828.py, the two UKF-guided defensive-TT/DMIS
runners, and run_c2_exact_likelihood_laplace_phase8b_20260904.py under
docs/benchmarks. The proposal-list builders require contextual review too.

The current generic UKF and three sampler primitives already expose fixed
signatures and JIT-on factories. Reuse those numerical authorities. Existing
stateless_random_tf and stateless_gamma_tf helpers preserve ordinary TensorFlow
seed words/conversions and are candidate building blocks, subject to actual
CPU/GPU context tests. The original three UKF adapter files did not all exist
at3582b4ac; freeze385a348b9 for this added caller repair rather than inventing
an unavailable original baseline. The prepared APF evaluator keeps its
September baseline in the separate core resource study.

Fingerprint comparison must bind each actual numerical/source payload. A
permitted rounding change can change a byte fingerprint; it must be reported
and independently recomputed, never stamped with the original identity. Route
settings, seed words, discrete ancestor/component selections and failure
statuses remain exact comparison requirements. Existing strict recomposition
and finite-difference checks retain their own bounds. A fuller unexecuted
implementation review is saved in/tmp/filter-repair-c2-preparation-design-notes.md.

Recovery review: the referenced ignored C2 fixture JSON is absent from both
current checkouts and Git385a348b9. Its exact seeded diagnostic generator and
fixture construction source remain available. Before the C2 numerical pilot,
regenerate the same n4/model52/observations42/T20 reference from that frozen
source, record its hashes and environment, and use identical frozen bytes for
all original/current arms. This is independent reference fixture recovery,
not permission to replace runtime random streams. A read-only Git archive of
the complete385a348b9 package is prepared at
/tmp/bayesfilter-c2-preparation-original-385a348b9 with713 file hashes.
The uninstalled test draft's source-loader API and unset program field are
incorrect; use real frozen source files for fingerprinting and bind the actual
public prepared evaluator. No draft is acceptance evidence.

The first execution allocation is a K=1 pilot after the frozen core cohort.
It is limited to24 serial workers,1800 CPU and3600 GPU process-seconds,
900 seconds per numerical worker and300 seconds for fixture/readback checks.
Register it as c2_preparation_k1_allocation with the completed core checkpoint
as its starting charge. Remaining compiler families require a subsequent
bounded allocation; this pilot cannot close the whole C2 gap.

Baseline385a348b9 is a complete materialized Git archive, verified against Git
by the isolated diagnostic loader; source fingerprints read the real frozen
files. Freeze full CPU and GPU originals before editing numerical sources:
T3/N16/seed9104, T4/N24/seed9102, T5/N16/seed9101, T3/N20/seed9103 with the
existing0.7 observation perturbation, plus T3/N16 negative and large seed
contexts(-9104 and4294976400). Record branch tensors, every proposal diagnostic,
manifest fields, actual branch/compiler/program IDs, value and analytical score.
Check invalid stationarity, initial/later nonfinite observations and small N,
preserving error class/message and first-failure ordering. Seed changes are
reference probes of the existing seed contract, not an RNG migration.

After source/fixture recovery, register these commands in the existing runner:

```sh
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py test --group c2_preparation_fixture_cpu --device CPU --test-timeout-seconds 300
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py test --group c2_preparation_k1_original_cpu --device CPU --test-timeout-seconds 900
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py test --group c2_preparation_k1_original_gpu --device GPU --test-gpu-index 3 --test-timeout-seconds 900
```

Qualification will compare the complete current public call against frozen
original records at the existing1e-10 FP64 bounds with exact discrete decisions;
existing phase1 strict recomposition/finite-difference limits remain unchanged.
Verify one retained trace, live observation/theta/seed operands, an enclosing
XLA While and no Python callbacks. Test shared full/prefix analytical evaluation
against original sliced-prefix programs. Keep public entry validation and
artifact construction as explicit host boundaries. General complex-eigenvalue
stability diagnostics are entry validation, not numerical time feedback.

Any changed fingerprint must bind actual current bytes; unchanged route,
classification and settings remain exact requirements. The cost pilot must
measure complete public preparation as well as the retained numerical owner,
three fresh paired blocks and128-call current reuse, with the same provenance,
contention, allocator/RSS, source freeze and exit rules as core resources.
Keep all originals/failures and stop at the first unexplained output/error/RNG
mismatch. Do not accept a syntax-only repair or extrapolate tiny-fixture capacity.
The draft under/tmp is uninstalled and unexecuted until the original snapshots
are preserved. Primary-agent review specifically caught the absent fixture,
invalid draft loader API and unset evaluator field before any C2 numerical run.

Pilot progress:05401 recovers the original seeded diagnostic fixture with
SHA2562957a6faeaaea0de893b010a5fd8d66b5e1fae82fb75e0be1645a2524dde603c.
05402/05403 freeze six complete original cases per CPU/GPU plus four error
contracts before numerical edits.05404 is the first passing K=1 CPU full-call
witness.05405/05406 each pass13 checks: complete records, live inputs, enclosing
XLA, original error ordering, shared prefix/default score and existing phase1
checks. Maximum CPU/GPU absolute errors1.208e-13/5.684e-14.

05407 preserves159 passes/two preflight failures. GPU groups had explicit GPU
launch arguments but lacked the runner's matrix-device registry; register all
C2 GPU groups. The graph-size test assumed exact total-node equality.05408
preserves the diagnostic failure: T3/T7/T11/T3 have2630/2632/2632/2630 nodes,
all nine functions; only one Fill and one Const are added. TensorFlow changes
large static-zero tensor construction from Const to Fill. The corrected check
requires identical computational-op inventories, stable T7/T11 graphs and
repeat T3 graphs. This is a harness representation correction; no numerical or
execution allowance changes. Runtime sources remain the ones qualified05405/6.

05409 passes161 graph-size/policy/runner checks. The matched K=1 cost driver is
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/run_c2_k1_resource_matrix.py,
using the registered c2_preparation_k1_cost_{original,graph,xla}_gpu groups
and three counterbalanced --repeat blocks. Each public boundary gets20 warm
calls; current XLA also gets128 fixed-input public reuses. The driver enforces
source freeze, the24-worker/1800-CPU/3600-GPU allocation and first bad-block
stop. Terminal c2_preparation_k1_terminal_cpu verifies full records, provenance,
process exit, same physical GPU and the unchanged comparison bounds.

K=1 unit completed through05419; numerical, API, execution and matched resource
checks pass. See filter_gradient_c2_k1_preparation_result_20261001.md.
Fixed/defensive mixture and other C2 preparation families remain open.
