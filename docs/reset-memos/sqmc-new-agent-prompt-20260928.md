# Prompt for a fresh SQMC agent

Open `/home/chakwong/BayesFilter-SQMC` in VS Code and create a **new** Codex
conversation there. Select the permission mode you intend to grant in the
conversation's permissions menu. For the broad access discussed previously,
the installed extension calls it **Full access**, with confirmation
**Turn on Full Access**. Managed restrictions may still apply; the agent must
check its effective environment. A subagent or a resumed old conversation does
not necessarily receive the new workspace permissions.

The prompt below expressly restarts the expired calendar window. Sending it
authorizes that bounded renewal; this document alone does not change the old
budget or any sandbox setting. It does not increase the original 12 GPU-hour
total, including already-used compute.

Paste the following into the new conversation:

```text
Continue the SQMC work in /home/chakwong/BayesFilter-SQMC on branch
sqmc-development. First read the applicable AGENTS.md and CLAUDE.md, then:

docs/reset-memos/sqmc-development-checkpoint-20260928.md
docs/reset-memos/sqmc-development-handoff-20260928.md
docs/plans/sqmc-expanded-comparison-20260926.md

Verify the checkout, branch, uncommitted changes and effective writable roots
before editing or launching anything. Preserve all existing work and evidence.
Do not resume the old conversation or re-run completed research by default.

Complete the requested expanded Kalman comparison: P44 d=3 at T=10,120;
full A and full SPD Q at d=3 and d=10, each at T=2,10,120. Use the four planned
routes, at least 1,000 particles, all score coordinates, actual scores and
absolute errors. Finish the runner audit and bounded GPU checks first.

I authorize a new 14-hour elapsed execution window starting with this resumed
campaign's first numerical check. Keep the original aggregate 12 GPU-hour
cap inclusive of all earlier pilots, checks and retries, and the original
limit of two infrastructure retries per unit. Reconcile actual prior use and
record the remaining budget before launch; do not silently expand compute.

Continue necessary local repairs and execution within that scope without
repeated procedural confirmation. Respect actual platform permissions and
ask only for a genuine unresolved boundary. No package/environment changes,
merge, push, publication, HMC run or scientific/default promotion is requested.

Keep the active checkpoint short. Preserve complete logs and versioned results,
and explain invalid candidates separately from infrastructure failures. Finish
with the detailed Kalman comparisons, a material-changes summary, uncertainty
and limitations, and a terminal review. A passing smoke test is not proof of
score accuracy or method superiority.
```

