# q=20 72-core process-parallel canary attempt 04

Date: 2026-09-03  
Controlling plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Artifact root: `docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/canary/attempt-04/`  
Status: `PASS_CANARY_FULL_RUN_AUTHORIZED`

The fresh post-repair canary completed in `585.6644140318967` seconds, below
the `1,200` second cap.  It exercised the requested sequential barriers:
`8x4` screen workers, `2x8` selection workers, and `6x4` finalization workers.
Every declared task produced one durable record, all worker affinity sets were
disjoint within their barrier, CUDA was hidden before TensorFlow import, and
workers reported XLA enabled with the expected q=20 target identity.

The fixed-seed serial reference and process worker agreed for samples, target
log probabilities, log-acceptance traces, and target scores at the predeclared
`1e-9`/`1e-8` tolerances.  The machine-readable source of truth is
`.../canary/attempt-04/canary_summary.json`; the run manifest records the exact
command, environment, commit, source hashes, CPU IDs, and cap.

This is an engineering/mechanics admission only.  Acceptance, timing, and RSS
remain descriptive.  The historical fixture does not provide fresh tuning
evidence, and this canary does not establish whitening, mode discovery,
convergence, posterior correctness, sampler ranking, CPU default status, GPU
speedup, or Phase 9B readiness.

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Admit full staged diagnostic | topology, identity, finite/status, durable coverage, serial/process parity | passed | prepare fresh charts and run P3 barriers | no posterior or convergence claim |
| Keep route diagnostic | repository GPU-default and scientific gates | CPU exception explicitly labeled | preserve GPU canonical policy | no CPU default or speedup claim |

| Inference class | Status |
|---|---|
| Hard veto screen | Passed canary engineering checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Barrier wall time, worker RSS, and acceptance |
| Default readiness | Not assessed; GPU default unchanged |
| Next evidence needed | Full staged diagnostic and independent numerical/posterior validation |
