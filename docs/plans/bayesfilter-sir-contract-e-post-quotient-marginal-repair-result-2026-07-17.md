# SIR Contract E Post-Quotient Marginal Repair Result

Date: 2026-07-17

Status: `PASS_ENGINEERING_REPAIR`

The two-step Sinkhorn truncation was a bug.  The registered SIR Contract E
route now uses 20 fixed annealing warm-start iterations followed by 100 fixed
terminal-epsilon IPFP iterations.  The actual row-quotient plan consumed by
Contract E is gated on both marginals using a float roundoff model, and the
total JVP/VJP differentiates the same finite balancing program.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the localized transport repair | All 16 original audit seeds satisfy both final consumed-plan marginals at roundoff scale | No non-finite value/score, XLA, identity, Cholesky, or derivative veto | Longer horizons and LEDH--teacher agreement were not run | Resume the predeclared `T=1,2,5` comparison ladder without retuning the transport schedule | Filtering accuracy, teacher agreement, HMC readiness, leaderboard readiness, or all-model correctness |

## Evidence

- Independent `N=256,d=18,T=2` GPU/XLA design seeds `88100--88115`: 16/16
  valid, maximum row residual `5.11e-15`, maximum final column residual
  `3.55e-14`, minimum covariance-gap eigenvalue `0.48388`.
- Final-only audit seeds `87200--87215`: 16/16 valid, maximum row residual
  `5.00e-15`, maximum final column residual `6.66e-15`, minimum covariance-gap
  eigenvalue `0.49230`, and 16/16 finite scores.  The previously invalid seeds
  `87202`, `87203`, and `87215` are repaired.
- Full repaired SIR scalar: manual score versus autodiff maximum absolute error
  `4.44e-16`; manual score versus same-scalar central finite difference maximum
  absolute error `2.33e-11`.
- The complete 100-step manual VJP compiled under CPU XLA and matched its eager
  result to maximum absolute error `3.33e-16`.
- Trusted RTX 4080 SUPER GPU/XLA execution used a hard 8192 MiB TensorFlow cap;
  the production-shaped canary peaked at `31,356,672` allocator bytes.

Structured evidence is in
`docs/benchmarks/artifacts/sir_contract_e_post_quotient_marginal_repair_20260717/result.json`.

## Test Status

- Focused final repair tests: `4 passed`.
- Registered CPU/XLA and identity checks: `8 passed`.
- Affected suite: `63 passed`; one unrelated frozen pre-edit baseline test
  remained failed (`-71.3687108484` observed versus `-71.4006261175` frozen).
  The historical `balance_steps=0` transport value was separately verified
  bitwise unchanged, so this failure is not attributed to the repair.

## Post-Run Red Team

The strongest alternative explanation is that `100` balance steps work only on
the tested `T=2` domains.  This would not invalidate the localized bug repair,
but it would veto longer-horizon promotion.  A `T=5` marginal failure, a
same-scalar derivative failure, or a nonpositive covariance gap under the
frozen schedule would overturn continuation.  No ranking or scientific
accuracy claim is supported by the present marginal evidence.
