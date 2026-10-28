# C2 scoped source binding for the factor-native consumer

28 October 2026. Continuation after commit 5c97d364c. Runtime classification:
standard-library source/provenance glue around the existing TF/TFP target;
CPU-only integration diagnostics. No mathematical, numerical, sampler or
posterior-policy change is proposed.

## Question, baseline and pass criterion

Can the public Phase 9 builder consume the clean C1-compatible checkout with
an explicit source binding, without monkeypatching global pins or mistaking a
factor-native score for the historical covariance-score implementation?

The baseline public loader pins the historical root and commit while its
provenance reports a new, unadmitted factor-native symbol. C2 preview passes
33 integration and 36 core tests only with process-local pin injection. The
candidate is f9291ee4437aa8c00dabbe331bffc6871b574c5c in
/tmp/BayesFilter-modern-hmc-c2-exact-source-20261028, a linked worktree of
/home/ubuntu/python/BayesFilter, not a replacement filtering implementation.

Pass requires a public builder call with explicit root and expected commit,
the real clean-source audit, checked import origins, provenance issued from
the actual factor-score callable, and independent numerical-reference parity.
Validation must reject stale commit, dirty source, foreign checkout/import,
wrong score symbol, pending provenance and payload/root mismatches. Default
legacy consumers retain their current pin and validator. A payload cannot
authorize itself: its validator needs the caller's explicit expected binding.

This qualifies only source binding and numerical compatibility for the tested
CPU target. It cannot authorize the consumer master, whose historical P19
artifact is missing, or establish posterior convergence, empirical inference,
GPU/full-chain readiness, method ranking or release. R-hat remains reporting
only during tuning; no tuning run is part of this step.

## Implementation and verification

1. Add a small standard-library binding module to MacroFinance. Verify the
   supplied root belongs to the BayesFilter Git repository, exact expected
   commit and clean state. Check existing/imported BayesFilter modules belong
   to that root and are tracked; identify the actual exported factor-score
   callable and record its source hash. No registry, token or new launch ledger.
2. Add an optional explicit binding to Phase 9's public builder/loader. Preserve
   the legacy default and diagnostic route. Produce factor-native provenance
   only after loading and checking the real callable. Add opt-in validation to
   the existing governance validator; unrelated consumers do not gain implicit
   acceptance of the successor schema.
3. Exercise actual Git fixtures for positive/negative source checks and run a
   public-builder integration against the clean C1 candidate, preserving the
   existing value/score and row-isolation tolerances. Run the unchanged prior
   integration matrix for regression. Use a temporary consumer overlay first;
   apply only the tested delta to current external files after hash/diff checks
   and required platform permission. Never commit their pre-existing changes.
4. Preserve the exact patch, source hashes, tests, commands and measured cost;
   refresh the master with the achieved scope and remaining consumer work.

Commands use /home/ubuntu/anaconda3/envs/tfgpu/bin/python, CPU deliberately hidden
with CUDA_VISIBLE_DEVICES=-1, TF_FORCE_GPU_ALLOW_GROWTH=true, bytecode disabled,
pytest plugin autoload disabled, four intra-op/two inter-op/OMP four threads.
Run focused tests with the test directory as confcutdir and cache disabled.
Use scripts/diagnostics/run_modern_hmc_c2_source_preview.py for regression and
a separate public-binding integration driver with no global-pin injection.
Output root: N5/c2-scoped-binding-20261028-01, fresh attempt directories only.
The 1,800-second test ceiling is inherited operational headroom for compiler
startup, not a numerical criterion; timeout preserves failure and triggers
diagnosis. No new allocation: deduct measured durations from the existing
upper-bound balance and leave separate SIR funds untouched.

## Default audit, pre-mortem and skeptical review

The selected commit is inherited from C1 byte equality and C2 compatibility,
not inferred from current HEAD. Clean-tree checks are inherited reproducibility
requirements; they do not certify scientific correctness. A linked worktree is
an explicit deployment option of the same BayesFilter repository. The old root
and global pin remain defaults; no automatic fallback or global source switch.

The chief misleading success would be checking one checkout but importing
another, or accepting a caller-stamped symbol without checking its callable.
Tests must expose both before numerical integration. Missing P19 evidence is
a real consumer-program blocker, but does not invalidate a BayesFilter-owned
source-binding test. Source drift, invalid numerical parity or import leakage
invalidates the attempt and triggers local repair; failed source fixtures are
expected test outcomes. No confirmation run is launched from compatibility.

Skeptical audit passes for this narrow engineering repair. It changes an
explicit opt-in API and tests both old rejection and new scoped validation.
It does not weaken the global validator or silently admit Phase 11/13 retained
campaigns. MacroFinance AGENTS.md and relevant memory sections were inspected;
all active numerical callbacks remain BayesFilter-owned. The missing master
evidence is preserved, not bypassed or fabricated.

## Execution review

The isolated source-binding tests pass 21 checks. The first public and regression
attempts failed because the temporary consumer overlay omitted the unchanged
Phase 8 synthetic-data helper. This is a harness import failure; preserve both
attempts and charge their durations. Copy and hash-bind that fixture dependency
before retrying in fresh directories. The corrected public API test and all
33 existing numerical/row-isolation regressions pass. The installed delta also
passes 21 source tests and the direct public-builder check without pin injection.

Post-install review found source provenance was unnecessarily inserted into
the GPU-policy dictionary, which contributes to the target manifest. Move it
to the adapter's dedicated provenance record. A new regression requires that
explicit binding and the same diagnostic runtime produce equal target
signatures, with GPU readiness remaining false. Recheck before installing this
small correction; no sampler or historical target identity is rewritten.
