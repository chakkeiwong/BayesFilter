# q=20 72-core process-parallel canary attempt 05

Date: 2026-09-03  
Controlling plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Artifact root: `docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/canary/attempt-05/`  
Status: `PASS_CANARY_FULL_RUN_AUTHORIZED`

This post-repair canary completed in `587.6819816830102` seconds, below the
`1,200` second cap.  It exercised the sequential `8x4` screen, `2x8`
selection, and `6x4` scope-finalization barriers.  All expected tasks were
durable, workers used disjoint assigned logical CPUs, CUDA was hidden before
TensorFlow import, and every worker reported XLA enabled with the expected
q=20 target identity.

The fixed-seed serial/process comparison passed for samples, target log
probabilities, log-acceptance traces, and target scores at `1e-9` tolerances
for samples/values and `1e-8` for scores.  The strict artifact writer now
preserves any non-finite diagnostic as a tagged value, so a candidate-local
numerical failure cannot erase its typed result row.  The machine-readable
source of truth is `.../canary/attempt-05/canary_summary.json`.

This is an engineering/mechanics gate only.  Acceptance, elapsed time, RSS,
and any candidate diagnostics are descriptive.  It establishes no whitening,
mode discovery, convergence, posterior correctness, sampler ranking, CPU
default, GPU speedup, high-dimensional scaling, or Phase 9B readiness claim.

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Admit fresh full staged diagnostic | topology, identity, finite/status handling, durable coverage, serial/process parity | passed | prepare fresh charts and run P3 barriers | no posterior or convergence claim |
| Keep route diagnostic | repository GPU-default and scientific gates | CPU exception explicitly labeled | preserve GPU canonical policy | no CPU default or speedup claim |

| Inference class | Status |
|---|---|
| Hard veto screen | Passed canary engineering checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Barrier wall time, worker RSS, and acceptance |
| Default readiness | Not assessed; GPU default unchanged |
| Next evidence needed | Full staged diagnostic and independent numerical/posterior validation |
