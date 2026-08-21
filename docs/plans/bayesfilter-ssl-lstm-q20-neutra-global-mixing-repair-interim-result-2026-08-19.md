# SSL-LSTM q=20 NeuTra global-mixing repair interim result (2026-08-19)

Plan: `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-plan-2026-08-19.md`

## Outcome

The circular global-archive prerequisite has been removed and the corrected
NeuTra workflow is executable through the exact transformed-target boundary.
The trusted GPU training/HMC canary has not run because the execution gateway
rejected the escalated launch with a model-availability `404`; a normal
managed-session attempt then failed closed, as required, because CUDA devices
were hidden.  Neither event is scientific evidence about NeuTra.

Completed evidence:

- Mode-specific starts are now diagnostic only.  A new anti-pooling gate
  rejects balanced pooled draws when individual chains remain mode-locked.
- A CPU/XLA weighted-forward-KL smoke consumed a batch of 32 target-query rows,
  all of which were target-valid and finite.  Proposal-weight ESS fraction was
  `0.505056`; one optimizer update and validation were finite.
- The freshly updated map was bound to the exact q=20 target.  At both known
  representatives, transformed values and explicit pullback scores were finite,
  target status was valid, and
  `log pi_z - log pi_theta(T(z)) - log|det J_T|` had maximum absolute residual
  `3.3584e-15` against the declared `1e-10` engineering tolerance.
- The GPU runner is prepared to use the eight existing terminal annealed-SMC
  populations as **weighted optimization replay only**: six independent banks
  for training, one for selection, and one untouched audit bank.  It does not
  call those rows a posterior archive or impose their weights on HMC.

No posterior, HMC convergence, global-mixing, mode-weight, or predictive claim
is supported yet.

## Claimed target and quantity computed

| Item | Classification |
|---|---|
| Claimed target | One exact NeuTra pullback HMC target whose retained chains forget initialization and traverse every material known mode. |
| Quantity computed so far | One CPU/XLA weighted transport update plus target/Jacobian/score adapter parity at the two known representatives. |
| Relation | Correct engineering prerequisite; different from and insufficient for the claimed global sampler target. |
| Source anchors | q=20 target `9a86e6...7278`, adapter `a8be6c...166f3`, checked geometry artifact, new plan and versioned replay-canary artifact. |
| Still unproved | Training convergence, proposal completeness, global HMC mixing, full-posterior correctness, mode weights, and posterior-predictive equivalence. |

## Mathematical decision

If separate chains remain in regions `A_j`, equal concatenation converges to
`J^{-1} sum_j pi(. | A_j)`, not generally to `pi`.  Consequently, balanced
initialization or pooled occupancy cannot supply posterior mode weights.  The
new coverage diagnostic requires every retained chain to visit every declared
region and to transition after initialization.  This remains a coverage veto,
not a standalone convergence proof; modern R-hat/ESS and numerical/status gates
remain mandatory.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Remove physical global archive as NeuTra prerequisite | Circular dependency identified and superseding plan recorded | No engineering veto | Independent posterior authority remains unavailable | Train from target-weighted replay and validate with exact HMC | Posterior correctness |
| Admit anti-pooling diagnostic | Mode-locked balanced chains fail; crossing chains pass coverage mechanics | Focused tests pass | Finite transition counts need HMC-scale MCSE/R-hat/ESS | Apply diagnostic to the one common pullback HMC run | Convergence from the diagnostic alone |
| Admit CPU/XLA replay smoke | 32/32 valid rows; finite update/validation; ESS fraction `0.5051` | No finite/status/adapter veto | Tiny batch and one update | Proceed to target-specific GPU/XLA capacity screen | Transport quality or global coverage |
| Admit exact pullback wiring | Value identity residual `3.36e-15`; finite values/scores/status | No target/Jacobian wiring veto | None at the two checked points; tails remain unchecked | Preserve parity check in GPU canary | Global geometry or HMC validity |
| Admit GPU training/HMC result | Not run | Infrastructure boundary blocks execution | Trusted GPU launch availability | Re-launch exact prepared command after explicit permission/gateway recovery | Any scientific result from failed launch attempts |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | CPU target/status/map-parity screens pass; no scientific GPU candidate exists to screen. |
| Viable candidates | The target-weighted replay workflow remains viable; no trained SSL-LSTM candidate is admitted. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Replay ESS, one-step loss/gradient, runtime, and adapter residual. |
| Default readiness | Not assessed. |
| Next evidence needed | GPU/XLA capacity/seed screen, untouched audit, then one exact common-kernel HMC global-mixing canary. |

## Run manifest

| Field | Value |
|---|---|
| Git commit recorded | Current `HEAD`; dirty concurrent worktree recorded by artifact |
| Environment | `tfgpu`; TensorFlow float64; CPU-only smoke with GPU intentionally hidden |
| XLA | Enabled; compile receipt emitted |
| CPU smoke command | `CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 python docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_replay_canary_2026_08_19.py --rows 32 --updates 1 --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r2` |
| Seeds | proposal `(20260819,1)`; transport `(20260819,2)` |
| Wall time | `12.5871 s` measured inside update/validation/pullback section; framework startup/compile adds launch wall |
| CPU artifact | `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r2/result.json` |
| Prepared GPU runner | `docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py` |
| GPU status | Hardware query showed both RTX 4080 SUPER devices healthy; trusted launch rejected by gateway before process creation; sandbox launch saw no CUDA and failed memory-policy check. |

## Post-run red team

The strongest alternative explanation for the clean smoke is that two local
Gaussian proposal components make the tiny replay unusually easy; this is why
the result is engineering-only.  The eight SMC banks also inherit known-region
support and cannot prove unknown-mode discovery.  Evidence that would overturn
the current positive engineering conclusion is a target/Jacobian mismatch on a
fresh held-out tail bank or a reproducible finite-state failure under the GPU
batch route.  Evidence needed for the scientific conclusion is cross-mode
mixing by every retained chain under the same exact pullback kernel, with modern
R-hat/ESS and downstream predictive diagnostics.

