# Replication approval settings — 27 September 2026

The owner requested fewer manual approval interruptions. Installed Codex:
0.153.4. Existing user configuration selected on-request approvals reviewed
by the user and workspace-write sandboxing. User rules already permitted
the tftwogpu interpreter and common inspection commands.

Two distinct problems were observed: the filesystem sandbox fails with
"error building bubblewrap command: mountinfo path is not absolute", and complex
inline shell commands are recorded as whole shell-wrapper approvals, so changing
their contents defeats reuse of an interpreter prefix. A saved rule is not
proof that every shell rendering matches it.

Installed, with elevated tool permission:
- /home/chakwong/BayesFilter/.codex/config.toml: project-scoped on-request plus
  approvals_reviewer="auto_review".
- /home/chakwong/BayesFilter/.codex/rules/replication.rules: eight exact prefixes
  for inspection and the current replication/diagnostic script entry points.

Both destination files were absent; installation refused to overwrite them.
Global config and earlier user rules were preserved. The copies here show the
installed contents. Eight positive and two negative command-policy queries
passed; rule-checks.json preserves the commands and responses. Negative queries
apply only to this new rule file, not to all existing broader user rules.

Auto-review automatically reviews eligible escalation requests; it does not
grant blanket access, and a reviewer may deny an action. Tool approvals are
separate from shell rules. The twelve locally configured MathDevMCP tool entries
already use approval_mode="approve"; no further tool policy was changed.

The live managed chat's permission mode cannot be changed by editing repo files.
Select Approve for me in the permission control (CLI: /permissions), or restart
and resume so the trusted project config/rules are loaded. If the app hides that
mode, its documented desktop setting is Settings > General > Permissions.
Live activation has NOT been verified by this agent. Organization/session
overrides may take precedence.

Execution convention: call the exact Python executable and stable script path;
put CUDA settings, timeouts and log handling in the script. Avoid heredocs,
inline shell pipelines, redirection and environment prefixes that prevent rule
reuse. Scientific approval is already governed by the owner's campaign request
and budget; a command rule does not expand that budget.

Do not set approval_policy="never" as a substitute for granting access: it
suppresses requests but does not enlarge the sandbox. Full access is a separate,
broader permission mode; it was not enabled by this change.

Official references checked:
- https://learn.chatgpt.com/docs/agent-configuration/rules
- https://learn.chatgpt.com/docs/config-file/config-reference
- https://learn.chatgpt.com/docs/sandboxing/auto-review
- https://learn.chatgpt.com/docs/permission-modes

Research checkpoint: docs/plans/iapf-kdm-replication-checkpoint.md.
