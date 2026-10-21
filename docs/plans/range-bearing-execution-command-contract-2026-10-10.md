# Range-bearing command and approval convention

Owner request: group necessary approvals upfront and execute the exact approved
forms, minimizing interruptions during authorized local research.

Installed rule file:
`/home/chakwong/.codex/rules/bayesfilter-range-bearing.rules`.
The installer validated 20 relative/absolute command forms and three negative
checks against this new bundle. This does not modify the global sandbox,
approval policy, existing rules, external tools, or scientific compute budget.
Existing rules still cover ordinary read-only inspection commands.

Canonical command prefix (same interpreter, flag order, and absolute script):

`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/scripts/<approved-script>.py <action>`

Approved scripts/actions:

| Script | Actions | Role |
|---|---|---|
| run_range_bearing_covariance_comparison.py | check, worker, run, report | Bounded diagnostic and frozen-tuning extension |
| finalize_range_bearing_comparison.py | checks, finalize, status | Fixed CPU-only checks, result writing, saved-result status |
| install_range_bearing_execution_rules.py | preview, install, verify | Fixed persistent-rule installation/verification |

Append supported options after the action. GPU commands still request escalated
execution, using the exact approved command. The master now launches its worker
with the same interpreter/-B/absolute-script convention. The fixed CPU wrapper
hides CUDA before any test import and preserves logs/manifests.

Use direct commands. Avoid leading environment assignments, heredocs, redirects,
shell wrappers, pipelines, alternate flag order, and broad Python/shell prefixes.
Put environment configuration, child launches, logs and artifact writing inside
the reviewed script. Existing arbitrary-interpreter permission was not broadened.

No connectors, Claude workers, network access, package mutation, destructive Git,
or publication are required for this completed scope, so no new permissions for
them were added. If a later authorized phase needs a new capability, first
prepare its concrete bounded command set, verify matching locally, then request
the missing permissions together before execution.

Persistent rule loading into a running session is platform controlled. A
successful local execpolicy check verifies matching, not a live-session reload.
Do not promise zero prompts or bypass platform approval. If one occurs, identify
the exact unmatched form or separate permission boundary and repair the command
shape before seeking additional scope.

Evidence: artifacts/range-bearing-execution-rules-20261010-01/
contains proposed rules and preview/install verification; the five synthetic
reporter checks and fixed reporting commands executed successfully afterward.
