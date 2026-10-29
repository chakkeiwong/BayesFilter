Update, 2026-09-28: the old running-process statements below are historical.
The sequential recovery launcher and fixed rules installer have been added to
the dedicated rules file (63 rules total). All five exact new command forms
passed codex execpolicy check. The installed-rule backup is
/home/chakwong/.codex/rules/bayesfilter-surrogate-hmc.rules.backup-20260927T200125187337Z.
Receipt: artifacts/surrogate-hmc-sequential-development-01/allow-list-verification.json.
Use the absolute tftwogpu interpreter followed by the absolute recovery script
and check/start/execute/status. No new broad interpreter or shell permission
was added; the previously approved interpreter covers the current session.
Tool rules do not authorize an expanded scientific compute allowance.
That separate budget decision is pending. See the sequential recovery plan.

# Surrogate-HMC command approvals — 2026-09-27

The owner requested reusable approvals for the authorized surrogate-HMC
campaign so routine execution does not demand repeated clicks. This is an
operational change; the scientific goal, promotion criteria, GPU policy,
approved compute budget, and running source snapshots remain unchanged.

The existing user configuration is approval_policy=on-request and
sandbox_mode=workspace-write. The existing default rules already allow the
absolute tftwogpu interpreter, common inspections, and GPU checks. A recent
process inspection used rg without the approved -n prefix, causing an
avoidable new-prefix request. The observed bubblewrap mountinfo failure also
requires trusted execution for otherwise routine commands on this session.

Updated user rule file:
/home/chakwong/.codex/rules/bayesfilter-surrogate-hmc.rules.
Its 61 rules cover repository inspection, device/process monitoring, pytest,
the closure diagnostic module, and 46 existing campaign/probe scripts using
both absolute and repository-relative launcher paths. No interpreter-wide
or shell-wide rule was added. The existing broad interpreter approval was
preserved. The user configuration was not changed.

Validation performed with codex execpolicy check only; no listed workload
was launched by these checks. All 110 positive command forms are allowed.
The dedicated file does not allow bare bash, bare python, an unapproved rg
prefix, or the incomplete git-diff prefix in four negative checks. The
three currently running process commands also independently pass against
the already existing default.rules, so their coverage does not depend on
loading this updated file.

Receipt and previous-file backup:
docs/plans/artifacts/surrogate-hmc-command-approvals-20260927-01/validation.json
and bayesfilter-surrogate-hmc.previous.rules in that directory.

Execution convention: use the exact conda interpreter and a direct script
command; set the checkout as workdir; use rg -n or rg --files; use git -C
with the exact checkout path and the documented inspection flags. Avoid
leading shell environment assignments, redirections, heredocs, or bash
wrappers that change rule matching. Configure CPU/GPU environment inside the
existing authorized launcher before framework import. Each pipeline segment
must independently match an allowed prefix.

File-based rules load at Codex startup. Use the already loaded approvals in
the current session and let the next normal Codex startup load this update;
the owner does not need to interrupt the running experiment to reload it.
Command rules govern shell execution, not all connector or application
approval surfaces. No connector write is needed for this campaign.

Official reference, checked 2026-09-27:
https://learn.chatgpt.com/docs/agent-configuration/rules

Current execution: initialization repair parent3477336/worker3477338
(session38896); retained supervisor3477801(session72288). The latter starts
the existing retained-sampling stage only after public qualification. The
master program remains incomplete and execution continues.
