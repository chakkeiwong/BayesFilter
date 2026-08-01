# P6 Result: Parameterized Spatial SIR

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `P6_COMPLETE_ONE_MEAN_LEVEL_NEUTRA_CONFIRMATION_TWO_PRECISE_BLOCKERS_CONTINUE_P7`

## Outcome

P6 has three honest terminal cell states:

| Cell | Terminal state | Binding evidence |
| --- | --- | --- |
| `SIR-SGQF` | `NEUTRA_CONFIRMED` | target `0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`; R4 result SHA-256 `e8b6c159648ade9f2919d97674ffc50a8b55d75d591a256291c3abfdcd4dbcce` |
| `SIR-UKF` | `IMPLEMENTATION_BLOCKED_GPU_SCORE_PARITY` | target-design GPU/CPU scale-normalized score gap `5.97e-7` exceeded the prospective `1e-7` limit |
| `SIR-ZC` | `TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE` | source audit established that parameter inference is an extension and retained-marginal/proposal-transport derivative closure is absent |

Only `SIR-SGQF` reached posterior identity, comparator, training, and NeuTra
confirmation. No scout or complete-data route was substituted for an
observed-data posterior.

## SIR-SGQF Evidence Ladder

| Rung | Result |
| --- | --- |
| Target design | CPU result SHA-256 `5d0d73f302b160b9f1277cd4ab5ef22ad53200f2c156cf6395d1e6a4ba0f9852`; GPU canary SHA-256 `51d61ea606521fe553555792ff771c1810424344bdcae2c300e42344731716b9` |
| Posterior identity | R1B GPU result SHA-256 `5cca9efae6147dbdcbd5ad12d0371451b58b6d26cc879ad1c267c0f40d100ea2` |
| Same-target comparator | R2 result SHA-256 `621c3d6e748eed38433efaa02ff097a971132de89f323f12702533723e3ce9b2` |
| Fresh 5,000-step GPU/XLA training | `dim3_lr1e3`; result SHA-256 `c69b4e4e02b68d13be74f7a87ffc0ec9b1d6a47bc8438d56c048577a78531854`; transport hash `dbd29efe786ec23c7b1098ba95ec6cad3a439b4889e04c67eeb2127965949c89` |
| NeuTra confirmation | R4 result SHA-256 `e8b6c159648ade9f2919d97674ffc50a8b55d75d591a256291c3abfdcd4dbcce` |

The comparator and NeuTra runs each used disjoint modern-R-hat tuning
verification, retained 2,000 warm-up draws per chain, and retained 4,000
posterior draws per chain. The NeuTra result had maximum modern R-hat
`1.0000689`, minimum bulk ESS `16,358.48`, minimum tail ESS `14,568.53`, and
passed the prospective simultaneous three-physical-mean agreement rule.

The supported claim is only same-target agreement of the physical posterior
means of `kappa`, `nu`, and observation-noise standard deviation for one T=20
fixture. It does not establish distributional equivalence or SGQF exactness.

## Target And Source Boundaries

The graph-native level-2 SGQF route computes the declared deterministic
approximate observed-data posterior and its total source-coordinate score. It
is not the exact nonlinear SIR posterior. The Zhao-Cui paper and author code
infer latent states with fixed rates in the SIR example; BayesFilter's three-
parameter posterior is an extension, not a reproduction. The local complete-
data density is wrong relative to a claim of a full observed-data posterior
and was never promoted.

`SIR-UKF` remains an implementation/device-parity blocker, not evidence that
the UKF mathematical target or research direction is invalid. `SIR-ZC` remains
a missing-target/derivative-closure blocker, not a failed NeuTra candidate.

## Repairs, Checks, And Budget

P6 repaired bounded harness and execution defects without changing target,
data, scientific criteria, hardware class, or phase budget. Material repairs
included target-design memory topology, XLA scheduling, diagnostic-mode
separation, GPU report completeness, JSON-normalized identity comparison,
plain-HMC tuning admission, optional condition-number telemetry, and fresh
training after the telemetry repair.

- Focused final CPU-hidden regression: `38 passed`.
- Every one of the 29 R4 recursive artifact hashes matched.
- Sixteen completed P6 manifests record `13,629.71` wall-seconds
  (`3.7860` hours) in aggregate. This is broad recorded wall-time accounting,
  not a pure GPU-utilization claim; failed pre-result attempts are disclosed in
  their repair records. No P6 budget veto fired.
- Git commit recorded by serious runs:
  `d269f5bbd8531b878d4f25897a357fbc8f172488` with a shared dirty-worktree
  disclosure.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| close P6 and continue P7 | one cell passed its complete ladder; two cells have precise terminal blockers | clear for SIR-SGQF; UKF parity and ZC target/source vetoes remain | fixture-specific, approximate-filter mean-level evidence | refresh and execute cross-cell integrity synthesis | SIR filter ranking/exactness, calibration, forecasting, broad robustness, production/default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Clear for the SIR-SGQF target-bound confirmation; supported for the two cell-local blockers. |
| Statistically supported ranking | None. |
| Descriptive-only differences | PF estimates, cross-filter values, losses, acceptance, runtime, quantiles, standard deviations, and correlations. |
| Default readiness | Not established. |
| Next evidence needed | P7 artifact/claim audit; separate prospective repairs for UKF device parity or Zhao-Cui extension design if those cells are resumed later. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | SIR-SGQF typed identity, CPU/GPU XLA replay, transport binding, GPU memory growth, batching, training parity, archives, and hashes passed; UKF device score parity remains blocked; ZC observed-data score route is absent. |
| Numerical/sampler validity | SIR-SGQF comparator and NeuTra runs passed health, modern R-hat, bulk/tail ESS, and sample-archive gates. |
| Scientific interpretation | supports only simultaneous agreement of three same-target physical posterior means for one approximate SGQF fixture; no cross-filter or epidemiological conclusion. |

## Post-Run Red Team And Drift Audit

The main risk is overreading an efficient, high-acceptance, highly converged
three-dimensional result as full posterior or scientific validation. The
prospective criterion was mean-level only, and P6 preserves that boundary.
Execution review also confirmed that short probes merely ordered kernels,
warm-up was retained but excluded from inference, retained sampling required a
joint convergence-and-agreement stop, and no UKF scout or Zhao-Cui complete-
data evidence crossed into the admitted SGQF identity.

