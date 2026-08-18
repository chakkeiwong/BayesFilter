# NeuTra Banana Matched-Kernel Cross-Over Result (2026-08-16)

## Outcome

The no-retuning bank-by-kernel cross-over completed in `64.85 s` on GPU 0.
It used the replayed seed-15, 6,000-update learned banana transport from the
discovery campaign and the exact verified kernels selected there. All 47
cross-over artifact hashes passed.

The result is decisive for the proposed causal distinction:

- Original bank with frozen central-selected `L=10`: passed.
- Central bank with frozen original-selected `L=5`: failed.

Together with the discovery diagonal cells, both banks pass at `L=10` and both
banks fail at `L=5`. The failure follows the kernel, not the initial-state
bank, for this frozen learned transport.

## Evidence Contract

| Item | Value |
|---|---|
| Discovery root | `docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3/` |
| Cross-over root | `docs/plans/artifacts/neutra-banana-hmc-matched-kernel-2026-08-16-r1/` |
| Learned state | Seed `15`, 6,000 updates, state hash replayed exactly from discovery |
| No-retuning rule | Exact `final_kernel_payload` copied from discovery; only bank changed |
| Bank cells | Original iid-normal bank and central deterministic bank |
| Kernel cells | Original-selected `L=5`, step `0.8361329642`; central-selected `L=10`, step `0.7709722546` |
| HMC gates | Shared sequential controller, four chains, warm-up/retained R-hat, finite health, movement, energy, and retained exact-law screens |
| Nonclaims | No default kernel, no default bank, no superiority, no SSL-LSTM transfer |

## Cross-Over Results

| Bank | Frozen kernel | Warm-up | Retained convergence | Retained exact-law | Status |
|---|---:|---:|---:|---:|---:|
| Original iid-normal | `L=10`, `0.7709723` | Pass | Pass at 2,000/chain, max R-hat `1.00308` | Pass | Passed |
| Central deterministic | `L=5`, `0.8361330` | Pass | Pass at 2,000/chain, max R-hat `1.00429` | Fail | Rejected |

The failed `L=5` cross-over reproduces the exact discovery failure: adjacent
cross moments 4 and 6 are `-0.03718` and `-0.03700`, with standardized
discrepancies `3.36` and `3.29`. Its HMC health remained finite and its
convergence gates passed, so acceptance/R-hat alone would have missed the
scientific failure. Native TFP divergence telemetry was unavailable and is not
interpreted as zero divergences.

## Mathematical Diagnosis

The analytic banana control passed with the original bank, and the learned
transport cross-over passes or fails according to the frozen leapfrog kernel.
The exact target, score binding, sequential controller, and initial-state bank
are therefore not sufficient explanations for the failed learned run.

The supported classification is **kernel-sensitive learned-transport HMC
dynamics** under the tested identity-z mass and trajectory grid. The shorter
`L=5` trajectory produces a systematic adjacent cross-moment distortion after
2,000 retained draws, while `L=10` passes the same checks under both banks.
This does not prove `L=10` is universally valid or that `L=5` is universally
invalid; it is target- and transport-specific evidence.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Start-bank hypothesis | Same kernel across banks | Rejected as primary cause | Only two banks tested, but both agree by kernel | Retain original bank; tune kernel against retained exact-law diagnostics | Universal start policy |
| Kernel hypothesis | Same bank across kernels | Supported for this transport | Only `L=5` and `L=10` were cross-over tested | Use the completed fixed-`L=10` confirmation as the bounded candidate | Universal `L=10` default |
| Controller/analytic target | Analytic original-bank positive control | Not implicated | Analytic map is not learned | Retain analytic arm as mechanics authority | Learned transport correctness |
| Banana HMC candidate | Retained exact-law agreement | `L=10` candidate passed bounded screen; `L=5` vetoed | One frozen learned state and bounded retained run | Longer confirmation with `L=10` under the unchanged gates | Production/default readiness |

| Evidence class | Status |
|---|---|
| Hard veto screen | `L=5` failed exact-law cross moments; `L=10` passed |
| Statistically supported ranking | None; no superiority claim |
| Descriptive-only differences | Acceptance, runtime, selected kernel, and R-hat progression |
| Default-readiness | Not supported |
| Next evidence needed | The fixed-`L=10` 5,000-draw confirmation is complete; next is downstream predictive-equivalence testing |

## Red-Team Note

The strongest alternative explanation is that the `L=10` success is a finite
retained-sample accident. The fixed-`L=10` confirmation is now complete under
the unchanged exact-law gate. The weakest evidence is any claim beyond this
frozen banana control; no setting should be transferred to SSL-LSTM.
