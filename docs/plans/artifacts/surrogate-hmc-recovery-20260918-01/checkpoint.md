# Surrogate-HMC recovery checkpoint — 2026-09-18

Active question: reliable parameter inference across repository models using
an imperfect full-DCTR force. Score mismatch alone is not a continuation veto.
Recovery stage complete; no model-inference candidate promoted.

Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917,
branch ledh-graceful-failure-execution-20260917, HEAD 8a4cffa2; merged main and
surrogate-hmc both 6fbcf314. Shared checkout read only. Preserve pre-existing
LG1 audit files and the execution-result edit. No new commit or GPU process.

Findings: original masters recovered; original revised program already restored.
Actual-source checks invalidate old toy MH PASS (joint energy, independent
acceptance) and alleged W2. Five shared-HMC mechanics tests pass. Old d=3
HMC runner omits prior, leaves two parameters inactive, disables corrections.
Old LGSSM SUCCESS was fixed-parameter cubature; later LG1 calibration failure
is not an inference result and persists in before/after containment replay.

Evidence: ../../ledh-surrogate-hmc-recovery-result-2026-09-18.md,
../../ledh-surrogate-hmc-model-coverage-2026-09-18.md, run-manifest.json,
recovery-provenance.json, toy-arithmetic-audit.json, shared-hmc-mechanics.json.
Full history and limitations are in these files, not this checkpoint.

Budget: containment charged GPU 1309.310489 s, CPU 3187.843224 s; three LG1
attempts exhausted. Separate closed audit charged GPU 243.392077 s and CPU
37.766647 s. Recovery's two permitted numerical checks used no GPU,
8.631416 CPU s / 8.708359 wall s; no further numerical check active or queued.
Old surrogate campaign's remaining balance unknown; do not pool budgets.
Conservative unused containment time, counting audit and recovery checks too:
63.5687 GPU hours / 15.1016 CPU-worker hours, without additional LG1 attempts.

Exact next action: owner decision on the proposed one-attempt LG1 inference
extension (<=2 GPU hours and <=1 CPU-worker hour inside the existing envelope),
then implement/verify the explicit DCTR-to-public-tuner consumer and forecast
cost before running. Plan: ../../ledh-surrogate-hmc-inference-continuation-2026-09-18.md.
Compare an exact Kalman endpoint with imperfect DCTR force first; separately
quantify frozen-DCTR posterior bias. Do not silently launch a fourth attempt.
