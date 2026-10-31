# M13 48-hour continuation: implementation and diagnostic review

Question: can the existing analytical DCTR proposal field support valid, converged
surrogate-HMC sampling when MH uses the separately frozen full-support
trajectory-importance endpoint? This is testing surrogate HMC. It is not a claim
that the field equals the endpoint gradient or that the frozen approximate
likelihood equals the original model likelihood.

The previous uniform small-step kernel was healthy but did not meet warmup
readiness. The completed seven-arm diagnostic showed that uniformly increasing
step sizes or trajectory length could stall one chain. Both fixed step mixtures
subsequently passed both discarded32-transition health probes. This supports
testing the planned mixtures; it does not prove better mixing or convergence.

Evidence root: artifacts/surrogate-hmc-m13-sequential-48h-development-01.
Fixed-step probe: mixing-probe-01/report.json.
Mixture probe: mixture-probe-01/report.json, guard-receipt.json, and
mixture-probe-source-check.json. The latter confirms source stability throughout
the GPU probe. Both components call the unchanged general neural-force HMC
kernel. GPU/XLA component parity passed. Equal independent fixed weights preserve
the mixture of endpoint-invariant kernels; the constant step chart has a
constant Jacobian that cancels in MH differences.

The master preserves the target/data/controls, modern Rhat and ESS requirements,
fresh-chain protocol, numerical vetoes and posterior materiality margins.
The six-look extension has a tighter simultaneous critical value and qualified
coverage under four reference controls. The frozen reference sources remain
byte-identical. Failed development harness attempts are preserved and charged.

Call-chain checks inspected: public consumer -> repository-issued mixture
binding -> public tune_hmc_kernel -> fresh sequential verification -> public
manifest reload -> public retained runner. The real Gaussian mechanics route
passed, including mixture identity and fresh seed propagation. Staged retention
continued to4000 cumulative draws in its mechanics test. Actual M13 preflight
must match the completed GPU probe binding and revalidate the reference.
Legacy identity behavior is covered separately.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Launch first scoped mixture | Diagnostic viability passed; posterior criterion pending | No probe health/source veto | Difficult chain may remain slow | Public tuning, warmup and fresh verification | Convergence |
| Enable six cumulative looks | Prospective uncertainty coverage passed | Reference source and estimator parity passed | MCSE assumptions remain approximate | Apply unchanged posterior margins at each look | Exact finite-sample coverage |
| Reserve second hypothesis | Both laws viable in short probe | Only healthy inconclusive first attempts retry | Neither law statistically ranked | Retry within32-hour aggregate tuning cap | A superior mixture |

| Inference status | Finding |
|---|---|
| Hard veto screen | Uniform large-step candidates failed movement at one start; both mixtures passed the bounded probes |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, squared jumps, runtime and old autocorrelation |
| Default-readiness | Not established; scoped optional candidate only |
| Next evidence needed | Fresh sequential verification, retained ESS/Rhat and MCSE-aware reference agreement |

Red-team finding: the proposal field differs substantially from the frozen
endpoint derivative at inspected states. Mixing the step scale may fail to
overcome that mismatch. A healthy short probe cannot overturn this risk; fresh
readiness and posterior comparison are the deciding evidence. This candidate
failure would not invalidate the fixed-endpoint surrogate-HMC principle.

Before launch,27 orchestration tests and12 scoped configuration/consumer tests
passed. The master executes its full focused suite, GPU qualification and actual
M13 preflight before scientific tuning. All development costs enter its ledger.

Approved command, from the repository root with trusted GPU permissions:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/scripts/run_surrogate_hmc_recovery.py start --allocation 48h --output /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-01 --cpu-budget 172800 --gpu-budget 172800
```

Hard deadline2026-09-30T16:27:39Z. Serial GPU workers; first tuning attempt<=20h,
aggregate tuning<=32h, at least10h reserved for retention, all bounded by actual
remaining CPU/GPU allowance and deadline. The master refreshes the next-phase
plan after each phase and each fresh/retained chunk. LG1 remains accepted;
M13 and the full all-model campaign are not yet complete.
