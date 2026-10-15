# Backward-guide integration checkpoint

Active task: commit the scoped surrogate-hmc work, integrate with main, and
synchronize the validated result to the intended target branch
`ledh-refactor-with-policy-fix` in
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
verified at `b28c25a6c749a08d191d05bd30a1ca5dfc0def69` with a clean worktree.
The prior implementation and checks used the wrong checkout. They do not
complete or validate the requested work on the LEDH branch.

Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, initial commit
`feddae400`. Preserve unrelated working-tree changes. The user now explicitly
authorizes a scoped commit and branch synchronization; no new push is requested.

Stage: wrong-checkout work preserved, intended-target integration pending.
On `surrogate-hmc` only, six propositions/proofs and pseudocode are in the
manuscript, an optional TensorFlow guide and shared-executor hook are implemented,
and all 10 focused tests pass. That PDF builds and new pages 54–58 were visually
inspected. MathDevMCP audits before and after are diagnostic, not certification;
the final audit covers all six propositions with unresolved formalization.
Main has a newer executor and already contains the earlier manuscript merge.
Merge the scoped changes while preserving the newer executor; do not copy whole
files or reuse the original checks as target evidence. Main is clean at
`023247142`; LEDH is clean at `b28c25a6c` and can subsequently fast-forward.

Contract: one finite likelihood program and its analytical total derivative;
retain physical importance weights, existing UKF/LEDH/OT/Contract E/GenUT path.
Use epsilon 1 and five Sinkhorn iterations only as the owner's development
comparator. No pilot fitting, model-score ranking, HMC, or canonical admission.
Existing local-flow log-Jacobian assumptions remain explicitly unverified.

Evidence: `docs/plans/ledh-backward-guide-integration-20261015.md`,
`docs/papers/ledh_younis_kdm_score/ledh_backward_guidance.tex`,
`bayesfilter/highdim/ledh_backward_guide_tf.py` and the narrow hook in
`bayesfilter/highdim/ledh_canonical_score_tf.py`.

Budget: up to 20 minutes CPU validation, three repair/test attempts, two bounded
MathDevMCP audit rounds plus targeted follow-ups. Two test attempts used: the
first failed collection because tf-gpu did not match the installed custom op;
the second passed in 11.56 seconds in the existing tftwogpu environment.
CPU checks deliberately hide GPUs before TensorFlow import; no GPU or pilot
training campaign is authorized by this scoped implementation plan.

Evidence: `result.md` records the equation/code/test correspondence, decision,
audit limitations and next research question. `run-manifest.json` records
commands, settings and source hashes. `pytest-attempt-2.log` records 10 passed
tests; `audit-after.md` is diagnostic-only. The three `compare-*.json` lexical
reports return mismatch, not semantic validation. All final build/render logs
are preserved here. Actual localized-flow Jacobian remains unchecked.

Next: check the remote, commit only the scoped changes on surrogate-hmc, and
merge them into the clean main worktree. Reconcile the proposal hook with the
newer shared executor and rerun applicable checks before fast-forwarding LEDH.
The skeptical synchronization audit is recorded in the plan. An additional
20-minute CPU validation budget and three localized repair attempts apply;
no scientific improvement or research-campaign authorization follows.
