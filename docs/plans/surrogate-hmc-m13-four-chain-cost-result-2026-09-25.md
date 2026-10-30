# M13 native four-chain surrogate-HMC connection

The public tuner's four-chain transition kernel successfully executed the
canonical analytical FP32 DCTR proposal force against the frozen independent
FP64 endpoint. All160 saved proposals were finite; every chain moved, with no
divergences, invalid endpoints or force fallbacks. There were150 acceptances
(93.75%). These40 transitions per chain are a mechanics diagnostic, not a
posterior sample on which to interpret R-hat or convergence.

At the four starting points, native batch4 and scalar canonical force values
agreed exactly. Endpoint values also agreed exactly with direct scalar
evaluation, and both force and endpoint replayed exactly. The expensive force
remained FP32/TF32/GPU/XLA; the small coordinate integrator and independent
endpoint used FP64 to satisfy the existing public interface. Controls remain
the declared unselected geometric .01 candidate on data1704, N4096/design9.

The original cost02 worker reached its850 CPU-second cap during the last L4
timing repeat. It remains EXECUTION_INCOMPLETE/JOB_CPU_BUDGET_LIMIT. Its completed
chunks were recovered by a separate CPU-only master stage, which checked donor
source hashes and revalidated every saved numerical/status tensor. No GPU
computation was repeated, and the absent timing repeat was not fabricated.

| Measurement | Result |
|---|---|
| Native four-row force, repeated call | 3.07359 seconds |
| Four-row independent endpoint, repeated call | .0780281 seconds |
| Four simultaneous chains, L2, repeated eight-step chunks | 9.04089 and8.95900 seconds per transition |
| Four simultaneous chains, L4, one repeated eight-step chunk | 15.43771 seconds per transition |
| First eight-step calls, including compilation | 133.521s atL2;182.735s atL4 |
| Largest absolute energy error in saved proposals | .610484 |
| Acceptance by chain over40 transitions | .825,.925,1,1 |

The force is the dominant observed cost in this diagnostic. This does not
establish efficiency relative to other samplers. A50% planning allowance gives
23.1566 seconds per four-chain L4 transition. Linear extrapolation toL8 gives
42.3470 seconds with the same allowance, but L8 was not measured. Neither is a
confidence bound; long-run runtime and selected trajectory length remain unknown.

Artifacts:
`docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-four-chain-cost-01`
(28 tests passed,1 failed; unsupported trace policy, no GPU launch),
`docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-four-chain-cost-02`
(29 tests passed; GPU mechanics and partial timing), and
`docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-four-chain-summary-01`
(28 tests passed; recovered result). Each executed pass preserves command,
environment, git/source provenance, status and resource accounting. Both GPU
donor and CPU recovery report unchanged sources during execution. The GPU worker
used escalated access and verified memory growth. CPU tests/reporting explicitly
hid GPUs. A preflight-only zero-GPU-budget rejection launched no worker or output;
the master now allows zeroGPU only through an explicit CPU-only budget option,
with regression coverage for exhausted positive budgets.

Charges: failed review7.681004CPU/0GPU; cost02 860.303842CPU/738.432726GPU;
recovery12.051444CPU/0GPU. Cumulative52625.494613292CPU/41652.696728097GPU.
Remaining1374.505386708CPU (22.91minutes) and185147.303271903GPU (51.43hours)
under the existing54000CPU/226800GPU caps.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Four-chain connection passed; timing partial | Actual canonical batch force through public affine kernel and native chunk loop | All saved chunks finite, valid, moving; no divergence/fallback | Controls unselected; tiny step; one L4 timing repeat absent | Scoped control protocol, public tuning, then retained posterior comparison, with measured budget | Posterior correctness, convergence, efficiency, all-model completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical veto in completed proposals; explicit resource interruption |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, energy and timing values from this short diagnostic |
| Default-readiness | No new numerical default; FP32 owner directive preserved |
| Next evidence needed | Scoped control selection; independently initialized public verification; retained MCSE/R-hat/ESS and reference agreement |

Post-run red-team: tiny steps can give high acceptance while exploration remains
poor. The diagnostic does not address that risk; adaptation and retained
posterior checks must. The weakest cost evidence is a single L4 repeat under one
hardware/load realization. Long-run timing, a different selected L, or recurrent
force fallback would overturn the forecast. The resource stop says the
reservation was too small for the planned diagnostic; it does not reject
surrogate HMC. The next work remains inside the master program.
