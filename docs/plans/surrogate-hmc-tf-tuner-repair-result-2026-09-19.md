# Public TensorFlow tuner repair result

The shared implementation blocker is repaired. Public `tune_hmc_kernel` accepts
`VerifiedTensorFlowHMCConfig`, which adds fresh dispersed-chain verification to
the existing TensorFlow nomination. The nomination's selected affine metric,
epsilon and L are frozen before discarded warmup and at least 1,000 verification
draws. Numerical/endpoint health, movement, energy, divergence and rank-normalized
split/folded R-hat <=1.01 govern handoff. Old diagnostic schemas retain no
authority. No runtime NumPy, private substitute tuner or admission-flag bypass
was introduced. Retained model-posterior comparisons remain unrun.

The earlier stop correctly blocked sampling without a tuning artifact, but
should not have stopped authorized implementation repair. Continuation uses
the existing campaign budget under the [repair plan](surrogate-hmc-tf-tuner-repair-2026-09-19.md).

## Checked implementation and evidence

- Corrected the rank-normalization denominator from `N-.25` to `N+.25`.
  For `p(r)=(r-.375)/(N+.25)`, ranks `r` and `N+1-r` sum to probability one;
  the previous denominator broke that symmetry. Independent SciPy and symmetry
  checks cover the change, along with stuck, separated and scale-separated chains.
- GPU Gaussian controls exercise the public endpoint-to-tuner-to-retained call
  chain with exact forces and with `g(q)=q+.15*(-q2,q1)`, an imperfect rotational
  field. Graph/XLA proposals agree exactly at checked states. Fresh verification,
  retained sampling and continuation pass. Independent mean/variance tolerances
  (`.15` and `.2`) are broad implementation vetoes, not statistical equivalence.
- Changed target identity, reused seeds, modified source hashes, relaxed R-hat
  metadata, changed decisions and tensor corruption are rejected. Host numerical
  guards run outside XLA so ignored graph assertions cannot grant handoff.
- Full LG1 N1024/T10 DCTR public-tuner XLA wiring passes, including diagnostic
  handoff refusal. Compile/first execution was 32.094 seconds and the next
  three-transition diagnostic 1.973 seconds. These are descriptive timings;
  they do not forecast a selected L or complete inference cost.
- 77 focused CPU tests pass, with three tests excluded by the no-pfor/long-moment
  diagnostic selection. The GPU control tests provide the retained moment check.
  CPU tests intentionally hid GPU devices. GPU jobs used trusted access to the
  RTX4080SUPER with verified memory growth.

Evidence roots under `docs/plans/artifacts/`:

| Root suffix | Outcome |
|---|---|
| `surrogate-hmc-all-models-20260919-tuner-repair-01` | CPU tests, logs and bounded launcher |
| `...-02` | 3.504-second failed script call; wrong keyword fixed |
| `...-03` | First successful complete Gaussian qualification |
| `...-04` | Added artifact/continuation validation passes |
| `...-05` | Full LG1 DCTR public-tuner XLA diagnostic passes |
| `...-06` | Final current-source Gaussian qualification passes |

Each GPU root contains command logs, run manifests, accounting and reports.
Roots 03–06 preserve executed source snapshots. Total campaign charge after
06 was 335.930 GPU seconds and 473.679 CPU-worker seconds before the final CPU
regression accounting. No LG1 inference attempt was consumed.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Shared tuner implementation qualified for scoped trials | Public call-chain, invariant and artifact checks pass | None in final checks | Target-specific tuning/mixing and implementation generality beyond tested controls | Prepare LG1 reference, scoped force selection and verified tuning | All-model posterior accuracy |
| LG1 inference pending | All four posterior comparisons NOT_RUN | Missing scoped prerequisites | Posterior error and full cost | Continue prerequisite work within campaign | DCTR or surrogate-HMC rejection |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Failed first script invocation repaired; no final qualification veto |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, acceptance and Gaussian moments |
| Default-readiness | Not established for any model or numerical controls |
| Next evidence needed | Scoped LG1 calibration, independent reference, fresh tuning and retained inference |

Red-team conclusion: Gaussian qualification could conceal target-specific
stiffness, poor geometry or slow mixing. The full LG1 wiring check closes only
compatibility, not those risks. Its scoped fresh verification and posterior
comparison must remain independent. Review was local; no new Claude review was
requested or claimed.
