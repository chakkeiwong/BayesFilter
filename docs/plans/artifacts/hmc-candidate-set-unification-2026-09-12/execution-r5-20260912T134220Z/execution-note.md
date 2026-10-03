# HMC candidate-set unification R5 execution note

Date: 2026-09-12
Plan: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`
Plan revision: R5
Execution root:
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/execution-r5-20260912T134220Z/`

This phase performs the bounded R5 repair identified by a skeptical source
audit of the R4 implementation. It does not run an HMC campaign, a GPU
process, a posterior comparison, or downstream MacroFinance/`dsge_hmc` code.

## Pre-execution audit

R4's focused tests were green, but the checked artifact boundary still trusted
some caller-rehashed fields. A relabelled inconclusive receipt, a verified
repair child with no repair action, or a reordered persisted work list could
therefore reach replay/resume validation. The typed bridge also needed its
non-authority and scope-identity behavior recorded explicitly. These are
engineering correctness issues, so the repair was limited to semantic artifact
validation, regression tests, documentation, and the existing dispatch guard.

## Changes executed

- Required candidate states and verified/viable ID lists to agree exactly.
- Required every verified member to have a passing, non-vetoed verification
  receipt tied to a persisted verification work item and matching mass/hash.
- Required every repair child to have one parent action, directional source
  receipt, exact `L`, and qualified execution/verification status.
- Validated persisted work-item order, candidate/family identity, cohort scope,
  unique verification attempts, and repair-action links; active `running`
  items are rejected as non-resumable artifacts.
- Reconciled charged, allocated, and released ledger units with reported used
  and reserved budgets.
- Required replay receipts to reference completed verification work, and
  rejected over-reserved budgets and viable IDs on shared-invalidity results.
- Added adversarial tests for inconclusive relabelling, missing repair lineage,
  and reordered persisted work.
- Declared the fixed-transport identity benchmark's legacy diagnostic policy so
  every fixed-transport config states its authority semantics explicitly.
- Made the Phase 9B checkpoint restore pass its serialized tuning policy
  explicitly, preserving the measured default for older payloads that omit it.
- Kept the typed ordinary/fixed-transport bridge explicitly non-authoritative
  until known-target, TensorFlow/XLA, resource, and active-consumer gates pass.
- Removed one redundant blank line so the unchanged dispatch contract remains
  under its existing source-size guard.

## Evidence contract

The primary criterion is semantic replay/resume integrity of the candidate-set
artifact. Hard vetoes are stale hashes, inconsistent state lists, invalid
receipt/work identity, invalid repair lineage, reordered work, and budget-ledger
inconsistency. Test outputs are mechanics evidence only. They do not establish
posterior convergence, target correctness, sampler superiority, XLA readiness,
or public numerical-default readiness.

## Commands

Environment: Python 3.13.13 in the repository TensorFlow environment. No GPU
process was launched; no HMC or posterior campaign was run. The focused
commands completed as follows:

```text
pytest -q tests/test_hmc_candidate_set_tuning.py tests/test_hmc_candidate_set_artifacts.py tests/test_hmc_candidate_set_adapters.py tests/test_hmc_tuning_dispatch.py tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py tests/test_fixed_transport_historical_caller_policy.py
79 passed, 2 TensorFlow Probability deprecation warnings

pytest -q tests/test_ssl_lstm_q20_phase9b_recovery_runtime.py tests/test_fixed_transport_hmc_binding.py tests/test_fixed_transport_hmc_tuning.py
122 passed, 297 gast/TensorFlow Probability deprecation warnings

python -m py_compile <touched Python modules and tests>
PASS
python scripts/inventory_hmc_tuning_routes.py --check
PASS; unclassified=(); stale_registry_entries=()
git diff --check
PASS
```

The earlier broader fixed-transport run was first blocked by two callers that
did not name `tuning_policy`; both were repaired in the smallest compatible
way and the affected policy and restore suites then passed.

## Remaining work

P2 known-target parity and TensorFlow/XLA qualification, P3 active numerical
consumer migration, and P5 default activation remain open. The existing public
ordinary and fixed-transport tuners retain their legacy selection authority
until those gates pass.
