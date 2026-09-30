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
