# Controlled-SMC / KDM active checkpoint

Updated 23 October 2026: owner disqualifies all previous N64 results.
Worktree: /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922.
Branch: ledh-refactor-with-policy-fix; HEAD 90a3acebd3965da5bdd64cb165dfdfe2d8cf0649.

Active task: explain the N1008 slowdown through the actual consumer.
Reviewed plan: [N1008 source and cost trace](ledh-sir-n1008-slowdown-trace-20261023.md).
No old N64 runtime, training, moment, score, or efficacy conclusion is admissible.
The master, primary old result notes and machine-readable disqualification notice
record this reset. No old result, weight or calibration is consumed by this run.

Stage: trace complete; report [slowdown cause](artifacts/ledh-sir-n1008-slowdown-trace-20261023/result.md)
and [fresh measurements](artifacts/ledh-sir-n1008-runtime-20261023/result.md).
Harness tests:8 passed. Fresh CPU-only data/source prepared in6.68 seconds.
The first tf-gpu import failed (Sylvester binary ABI); the corrected environment
is /home/chakwong/anaconda3/envs/tftwogpu/bin/python. No packages were changed.

Artifact root: docs/plans/artifacts/ledh-sir-n1008-runtime-20261023/run-02.
GPU attempt: attempt-01 complete; launch-01.log records progress. Three isolated source
workers: premerge, checked, current. N1008/T5/FP64/XLA, fresh paired inputs,
plain/conditional/physical arms, zero/e0 directions, 16 paired timing blocks.
All612 calls valid/repeatable; all24 cross-source parity checks pass. No retracing
or source drift. Current/pre-merge e0 ratios: plain1.2133, conditional1.3585,
physical0.7282; pointwise95% intervals exclude1, both datasets agree. Current versus
checked-repair grouping does not pass uniform speed non-harm. No trained guide
or native parameter batch is timed. Actual target GPU was idle at both run
boundaries; earlier busy preflight does not prove contention during measurement.

GPU wall509.35 seconds; resident-worker sum1327.79 seconds. All task workers
stopped, within2700s elapsed/8100s summed caps. No extra retry needed. The LaTeX
withdrawal notice is in the rebuilt PDF; no scientific efficacy claim was made.
The source/HLO trace is complete. It shows that the merged determinant-validity
and innovation-validity checks are live dependencies in the plain and beta=1
conditional analytical recurrences. The physical beta=.9 arm already needed
the determinant and benefits from the current 9x9 low-rank form replacing the
pre-merge 18x18 QR route. The intervention timings are not used for percentage
attribution because five XLA workers shared the target GPU. See
`artifacts/ledh-sir-n1008-slowdown-trace-20261023/attempt-01/`.
Exact next action: if repair is requested, design a guard-preserving kernel
optimization and validate it at N>=1008; do not remove the checks based on the
diagnostic variants.
Harness8 tests pass. Pre-merge/current warm calls and all diagnostic variants
complete; 24 healthy cross-source parity checks pass. Cap20min elapsed/100
summed worker-minutes, enforced by supervisor.
Production source and safeguards remain unchanged. Process-local interventions
are diagnostic only and are explicitly ineligible as runtime fixes.
Guide efficacy needs
fresh N>=1008 training rather than reusing disqualified weights/results.
Preserve unrelated changes. No commit or push requested for this task.
