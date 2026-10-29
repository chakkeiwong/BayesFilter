# Surrogate-HMC component campaign: additional 24 hours

User authorization: “you have 24 hours more”, 2026-10-07. Add 86,400 GPU-worker seconds to the original 86,400-second campaign ceiling: 172,800 seconds (48 hours) total, including all previously charged work. The extra authorization is not a new full-grid completion forecast.

The research question remains whether marginal-mixture weighting preserves LGSSM/KSC likelihood and analytical-score accuracy and reduces range-bearing errors on paired fresh datasets. The ultimate goal is testing surrogate HMC. This phase evaluates its likelihood/score components; it does not run posterior chains. The evidence contract, baselines, 4/8/16 dataset looks, 32 designs, independent references, per-scope controls, uncertainty rules, numerical vetoes and heuristic comparisons in [the active plan](surrogate-hmc-marginal-expanded-execution-2026-10-06.md) remain in force.

## Skeptical audit before implementation and execution

The active master holds its budget in memory and gives each worker a fixed deadline. Editing campaign.json while it is running would not extend that deadline and could be overwritten. Changing numerical or master source would invalidate the frozen source identity. A new output root that imports only completed scopes would discard resumable partial data. Therefore use a standard-library supervisor outside the frozen numerical source closure. Let the current invocation end normally, verify that its workers have stopped and its configuration and source still match, preserve its terminal state, increase only the declared budget, and invoke the unchanged master against the same campaign directory. Every new worker uses the master's existing fresh numbered attempt directory. Existing manifests and results remain unchanged.

The implementation audit confirms that the worker reloads frozen controls and timing calibration, references with matching data checksums, complete datasets, and completed paired rows. It recomputes only incomplete work; no parameter or method selection uses the interrupted claim data. Resumption therefore answers the same question without reusing partial results as if they were final. Recompilation across processes remains an existing FP32/TF32 repeatability limitation, explicitly preserved in the active report.

Primary promotion criteria and vetoes remain unchanged. A worker timeout is an infrastructure/budget limitation and remains unfinished; it is not numerical evidence against the method. The supervisor may resume only ordinary INCOMPLETE state. Engineering/precision/source invalidity, live or orphaned workers, or unexpected configuration changes require a repair and must not be bypassed. No CPU numerical fallback, additional GPU, or package/environment change is authorized by this extension.

Assumptions: 24 extra hours means an additional 86,400 worker seconds, not resetting consumed time to zero. The two-hour scope cap and three-attempt limit remain unchanged. The early diagnostic is a controller test that preserves source identity, consumed time, frozen controls and cached results while creating a fresh attempt and refusing live-worker/configuration/source mismatches. This is orchestration validation, not scientific evidence. The extension does not resolve the undefined nonlinear dimension variants or establish HMC correctness.

## Execution and artifacts

Controller: scripts/extend_surrogate_hmc_marginal_budget.py.
Active root: docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-04.
Extension evidence: budget-extensions/20261007-01/ under the active root.
The controller records the authorization, exact command, Git revision, original configuration, source snapshot, timestamps and terminal pre-extension campaign, and appends the budget history without changing older attempt manifests. It monitors the current master, launches the unchanged master after safe activation, and refreshes the existing result report. Trusted/escalated execution uses /home/chakwong/anaconda3/envs/tftwogpu/bin/python and the same RTX 4080 SUPER; GPU workers retain the verified memory-growth policy.

Exact invocation from the repository:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/extend_surrogate_hmc_marginal_budget.py --output docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-04 --additional-seconds 86400 --master-pid 147136 --reporter-pid 260380 --record docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-04/budget-extensions/20261007-01

Stop at the enlarged cumulative budget, completed defined grid, existing attempt limit, or a genuine engineering/source/precision continuation veto. Preserve and report missing coverage. New evidence will be interpreted per model, horizon, dimension and particle count; no global ranking or completed surrogate-HMC claim follows from a successful restart.

## Implementation review and validation

The controller only modifies campaign status, the cumulative budget and an appended budget-history record after the prior invocation exits. Frozen source identity and configuration are checked again at activation, and old attempt manifests, controls, references and scientific rows remain untouched. It stops only the verified reporting process for this campaign before taking over reporting. The unchanged master remains responsible for fresh attempt directories, saved-row resumption, worker deadlines and all scientific gates.

Twelve focused CPU-only orchestration tests passed on 2026-10-07, including an actual call through the unchanged master with simulated worker completion. They verify that completed jobs are skipped, a new numbered attempt is used for unfinished work, all consumed time and prior records are preserved, and live-worker, changed-source/configuration and numerical-veto boundaries are refused. Validation: artifacts/surrogate-hmc-marginal-expanded-20261006-04/budget-extension-checks-20261007-01/result.json and pytest.log. No scientific result or ranking is inferred from these tests.
