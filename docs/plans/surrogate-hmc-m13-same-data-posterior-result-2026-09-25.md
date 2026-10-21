# Same-data M13 posterior precision result — 2026-09-25

All16900 grid evaluations were valid. The fixed likelihood endpoint and the
independent higher-particle reference give very similar posterior moments on
the same data1704. Their means differ by3.0443e-5 for lambda_Q and5.6572e-6 for
lambda_R, or0.01018% and0.00339% of a posterior SD. Including3 reference MCSE,
the largest mean/SD discrepancy envelope is0.000362 posterior SD (0.0362%).
This is well below the prospective0.05SD local materiality screen.

The combined precision screen nevertheless requests grid refinement. The33/65
quadrature change reaches0.02663 posterior SD, exceeding its0.01SD allocation.
This is a reference-resolution repair trigger, not evidence of material error
in the endpoint and not a rejection of surrogate HMC. Both independent and
fixed likelihoods show almost identical grid shifts. Refining nested nodes
under the same likelihoods is the discriminating repair; changing likelihood
particles or surrogate controls would not address this finding.

| Quantity | Fixed endpoint | Independent reference | Reference MCSE |
|---|---:|---:|---:|
| Mean lambda_Q | 0.040527254 | 0.040557697 | 0.000025925 |
| Mean lambda_R | -0.131119517 | -0.131113860 | 0.000007930 |
| SD lambda_Q | 0.298952329 | 0.298964111 | 0.000003646 |
| SD lambda_R | 0.166799346 | 0.166805753 | 0.000001590 |

The reference uses N16384 and24 independent likelihood replicates; the fixed
endpoint uses N4096/8. Both grids use the declared prior, full wrapped law and
finite box +/-4 prior SD. Reference log evidence37.8277969 has MCSE0.0015570.
Reference probability beyond3 prior SD within the box is6.1611e-6. The minimum
importance ESS across all nodes was150.05 and largest weight0.06926. These are
explanatory; no unbounded-tail guarantee follows from them.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Endpoint discrepancy locally immaterial | All four envelopes below0.05SD | All nodes valid | Grid and omitted tails | Refine65 to129, reusing all existing nodes | Endpoint/posterior admission |
| Grid precision needs repair | Two components exceed0.01SD | No invalid numerical state | Whether65 is already sufficient | Nested reference refinement | Surrogate-HMC rejection |

| Inference status | Finding |
|---|---|
| Hard veto screen | All values/fit checks valid; source hashes unchanged |
| Statistically supported ranking | None |
| Descriptive-only differences | Posterior moment/grid differences and importance diagnostics |
| Default-readiness | No admission; FP32 force default unchanged |
| Next evidence needed | Nested resolution, domain sensitivity, scoped controls and public tuning |

Post-run red team: the coarse grid may alias the narrow observation-scale
posterior. Similar fixed/reference errors can cancel, so their agreement alone
does not resolve quadrature. Missing mass beyond the box remains separate.
The public tuner supports the typed position-only proposal binding; the small
mechanics callback is not yet a four-chain tuner consumer. Existing policy
permits FP64 small endpoint/integrator roles while preserving the costly FP32
canonical force; do not misclassify this as a requirement to convert the force.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-same-data-posterior --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-posterior-01`.
Plan: surrogate-hmc-m13-same-data-posterior-2026-09-25.md. The output root
preserves33 regression results, raw grids/rows, GPU memory growth, data checksum,
sources, commands, manifests and receipts. Charge1102.542987CPU/1093.406905GPU.
Cumulative48098.478077292CPU/37304.005855668GPU; remaining5901.521922708CPU/
189495.994144332GPU. Global budget unchanged. The next repair is inside the
existing master; no new direction or budget is requested.

## Nested129 refinement completed

The resolution repair passed: all49664 new likelihood evaluations were valid;
16900 prior evaluations were reused after source/data/hash checks. Fine129 vs65
grid envelopes are at most6.269e-6 posteriorSD (0.000627%), below0.01SD.
The endpoint/reference discrepancy envelope remains at most0.0003621SD,
below0.05SD. Mean differences are3.04611e-5 and5.65278e-6; reference means
are0.04055763045 and-0.13111435624, with SDs0.2989638811 and0.1668047331.
All32 focused CPU tests passed; master source snapshot unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Finite-box precision screen passed | Endpoint and grid margins passed | No invalid nodes or source changes | Omitted tails; importance approximation | Existing master: bounded domain extension, then scoped controls/public tuner | R² posterior qualification or chain convergence |

| Inference status | Finding |
|---|---|
| Hard veto screen | No supported veto in the finite-box diagnostic |
| Statistically supported ranking | None; no sampler ranking attempted |
| Descriptive-only differences | Runtime and individual weight diagnostics |
| Default-readiness | No new default; FP32 force remains intended target |
| Next evidence needed | Domain sensitivity and actual qualified HMC chains |

Strongest alternative explanation: shared importance limitations or posterior
mass outside the box could escape this check. Domain extension or qualified
chains with materially different moments would overturn broader reassurance.
The weakest evidence remains omitted-tail control.

Artifacts: surrogate-hmc-closure-20260922-m13-same-data-refine-01 under artifacts/.
Actual command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python
scripts/run_surrogate_hmc_program.py --stage closure --closure-stages
m13-same-data-refine --output
docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-refine-01.
Worker wall3085.44s; charged3106.309283CPU/3093.160567GPU.
Cumulative51204.787360CPU/40397.166423GPU; remaining2795.212640CPU/186402.833577GPU.
Same global54000CPU/226800GPU limits. Manifest records commit, environment,
GPU memory growth, seeds, data checksum and full command.
