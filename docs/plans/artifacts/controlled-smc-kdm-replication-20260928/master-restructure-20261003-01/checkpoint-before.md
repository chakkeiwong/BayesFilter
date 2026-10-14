# Controlled-SMC / KDM active checkpoint
Updated UTC 2026-10-02T15:06:10.406045+00:00. Owner authorized repair execution, 2026-10-02.
Checkout: /home/chakwong/BayesFilter; branch surrogate-hmc; base89065bc. Preserve unrelated work.
Master: controlled-smc-kdm-replication-master-program-20260928.md; dispatcher: scripts/run_controlled_smc_kdm.py.
Plan: docs/plans/controlled-smc-lorenz-repair-execution-20261002.md; result: controlled-smc-lorenz-repair-result-20261002.md.
Question: reproduce the adaptive recursion and reject unsupported Gaussian policy updates without confusing robustness extensions with source replication.
Checked:135 passes20 default bookkeeping checks; saved four-replica GPU recursion is bit-exact.138 passes39 diagnostic guard checks. Guards remain experimental:132/133 full healthy A/b parity failed.
Open:130 independent and135 CPU recursion disagree;136 equivalent Bellman arithmetic also fails.134 measures large-mean amplification and a time0 coefficient interpolation curve; no threshold selected.
Evidence correction:133 recursion-comparison fields are invalid due to one-versus-four-replica broadcasting;135 replaces them.137 was an import-only infrastructure failure, repaired in138.
Latest attempt: /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/attempt138_c2_lorenz_failure_rows; COMPLETE. A diagnostic PASS does not mean filter admission.
Next: Bounded repair stage closed: preserve accepted bookkeeping; next specify high-precision local recursion checks at12/11 and joint-update KL/model-adequacy calibration. Full recursion and untouched validation must pass before Figure6.
Repair attempts 9/9; charged seconds {"cpu": 6.094009807915427, "gpu": 176.24730683036614}; caps30CPU/300GPU. Nine includes one localized infrastructure retry; no resource increase.
Remaining current allocation {"cpu": 32.97093922289787, "gpu": 2998.5766364863957}; inherited {"cpu": 286.13787484405407, "gpu": 115949.87698720678}.
Running: []. Figure4 right complete; Figure6/Table2/KDM empirical incomplete; iAPF deferred.
