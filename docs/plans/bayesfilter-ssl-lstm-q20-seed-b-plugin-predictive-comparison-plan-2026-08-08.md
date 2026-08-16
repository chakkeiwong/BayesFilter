# SSL-LSTM q=20 seed-B plug-in predictive comparison plan (2026-08-08)

## Objective

Compare the ten-step predictive output distribution generated at the known
q=20 true free parameter with the distributions generated at two fixed
plug-in estimates from the already retained seed-B NeuTra HMC archive:
the pooled posterior mean and the coordinatewise posterior median.

This is the established plug-in procedure. The retained HMC draws are used
only to compute the two parameter summaries. They are not propagated as a
parameter mixture.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Does seed-B's estimated parameter produce an SSL-LSTM predictive output distribution close to the output distribution at the known generating parameter? |
| Candidate | Seed-B terminal checkpoint 4000, mapped retained model-coordinate draws from the r2 sequential archive. |
| Plug-in estimates | Pooled coordinatewise mean and pooled coordinatewise median of the 4,000 retained physical free-parameter draws. |
| True comparator | Locked q=20 `PRIOR_CENTER=(0.35,-0.08,0.65,0.05)` from the target source. |
| Predictive law | Current q=20 principal-root SSL-LSTM ten-step forecast, two replications per parameter row, common stateless Philox innovations across arms. |
| Primary output | Per-horizon predictive mean, variance, q05/q50/q95, and absolute/relative differences versus the true-parameter control. |
| Hard vetoes | Target/transport/archive identity mismatch; malformed or stale tensor receipt; nonfinite summary/forecast; invalid forecast status; visible GPU in the CPU/XLA route; repeated parameter rows not identical. |
| Explanatory diagnostics | Runtime, quantiles, path-level empirical distances, and mean-versus-median differences. |
| Must not conclude | Exact posterior correctness, mode-mass correctness, sampler superiority, model adequacy, or default readiness. |

## Evidence contract

- Exact target: q=20 principal-root target signature bound by
  `ssl_lstm_q20_neutra_seed_b_terminal.py`.
- Exact archive: two retained chunks, 1,000 transitions per chain, four
  chains; all warm-up chunks are excluded.
- Exact parameter summaries: compute after mapping every retained latent row
  through the frozen seed-B transport; no summary is computed in latent `z`.
- Exact forecast comparison: pass one parameter row and use 1,024 independent
  forecast-noise replications on the API replication axis. The parameter is
  therefore evaluated once per arm; only innovations vary.
- Common-random-number control: all three arms use the same forecast seed, so
  differences are attributable to parameter values rather than noise-bank
  identity. A separate independent seed is used for the material run and is
  recorded.
- Result artifact: versioned JSON and Markdown under
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-plugin-predictive-comparison-2026-08-08/r4/`.

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Mean and coordinatewise median | User-requested alternatives; primary mean, sensitivity median | Summary is mode-averaged or coordinatewise chart-dependent | Record both vectors and compare them |
| True vector | Source-locked `PRIOR_CENTER` | Wrong target chart or stale source | Target signature and source constant check |
| 1,024 forecast rows | Convenience diagnostic count, not a posterior sample count | Noisy distribution summaries | Record Monte Carlo role; no formal equivalence claim |
| Two forecast replications | Existing forecast API default | Wide path summaries | q05/q50/q95 and per-horizon variance are descriptive |
| CPU/XLA execution | Explicit reference/diagnostic exception | Slower than GPU route | GPU hidden before TensorFlow import; record XLA |

## Skeptical audit

Audit status: **passed after correction**.

- The first interpretation risk was propagating all 4,000 parameters. That
  computes a posterior-predictive mixture and answers a different question;
  this plan forbids it.
- The forecast API requires a static draw axis, so each plug-in parameter is
  repeated across rows solely to generate independent noise realizations. A
  repeated-row identity check prevents accidental parameter mixing.
- The true vector is read from the current q=20 target source and checked
  against the target signature; it is not inferred from retained draws.
- Mean and median are both predeclared. No post-run selection is allowed.
- This test is a predictive functional check, not an independent posterior
  authority. A pass cannot certify the parameter posterior.
- The run is CPU-only with XLA and hidden GPUs. No production code or model
  default is changed.

Execution repair: the first canary reached checkpoint restoration and found
that the archived trainer configuration predates two empty fields,
`fixed_output_scale` and `fixed_output_factor`. The lane-local loader was
updated to migrate only those absent empty fields, with no numerical transform
change. The retained archive and target remain unchanged.

Resource repair: the first material attempts repeated the fixed parameter over
1,024 draw rows, forcing the terminal filter to be recomputed for every row.
The repaired route uses one parameter row and 1,024 replication-axis noise
draws. This is the same plug-in predictive law and avoids parameter-row
recomputation; the forecast API still uses XLA and stateless Philox noise.

Artifact repair: the first canary artifact under `r1/` is preserved. The
optimized execution writes only under the new `r2/` root and never overwrites
the prior receipt.

Manifest repair: the completed `r2` receipt retained the old default value
`forecast_replication_count=2` even though its optimized material run used
1,024 replication-axis paths. The numerical output is valid, but `r2` is
archived as superseded; the final rerun writes the corrected manifest under
`r3/`.

Terminal review repair: `r3` still carried a stale repeated-row label and the
canary/material roots were not separated in code. The final `r4` run uses
canary seed `(20260808,81001)`, independent material seed
`(20260808,82001)`, one parameter row per arm, and 1,024 forecast-noise
replications. Earlier receipts are preserved as superseded engineering
evidence.

## Execution and stop conditions

1. Verify source, target, transport, archive manifest, all retained receipts,
   and tensor hashes.
2. Map retained `z` rows to physical free parameters and compute mean/median.
3. Run a 32-row XLA forecast canary for true/mean/median using the common seed.
4. Run the material 1,024-row plug-in comparison under a 900-second cap.
5. Write JSON and Markdown results. Stop on any hard veto or cap exhaustion.

The run must never read warm-up draws, modify retained artifacts, retune HMC,
or launch plain HMC.
