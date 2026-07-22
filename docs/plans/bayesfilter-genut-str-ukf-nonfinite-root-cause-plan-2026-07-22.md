# STR-UKF GenUT Non-Finite Root-Cause Plan

Date: 2026-07-22
Status: `DIAGNOSTIC_EXECUTION_AUTHORIZED`

## Research intent ledger

| Field | Frozen decision |
|---|---|
| Main question | Where does the first non-finite quantity arise in the `STR-UKF` GenUT `N=1002,T=100` path, and which mathematical operation causes it? |
| Candidate under diagnosis | The already-selected `epsilon=4`, `sinkhorn_steps=4`, `ridge=1e-6` FP32/TF32 GPU/XLA route. No control is changed in the reproduction pass. |
| Seeds | Reuse consumed failed-claim seeds `2026072291,...,2026072298` for diagnosis only. These seeds are ineligible for a future untouched claim. |
| Expected failure modes | State/tangent growth in the quadratic transition; `0*Inf` in likelihood-weight tangents; small Sinkhorn denominators amplifying JVPs; ill-conditioned Contract-E Cholesky JVP; or FP32 accumulation overflow. |
| Diagnostic criterion | Reproduce the failure and identify the first seed, time, stage, tensor class, coordinate, and relevant scale/conditioning statistic. |
| Diagnostic veto | Wrong controls/data/seeds, structural residual failure before the implicated operation, missing per-time increments, CPU fallback, non-XLA execution, or inability to distinguish value from tangent failure. |
| Continuation rule | First run an increment-only reproduction. Instrument only the first implicated stage and its immediate upstream inputs. |
| Not concluded | No repaired claim, no leaderboard admission, no new tuning choice, no superiority, and no default/HMC readiness. |

## Exact path to trace

For each time step:

1. optional structural transition and its manual tangent;
2. structural residual before reset;
3. Gaussian observation log density and tangent;
4. log-sum-exp likelihood increment and normalized weight/JVP;
5. Sinkhorn cost scaling, kernel, denominators, coupling and barycentric JVP;
6. Contract-E weighted moments, three Cholesky factors, affine restoration and
   their manual JVPs;
7. restored particles/tangents and recursive value/score accumulation.

The first pass records `value_increments` and `score_increments` already emitted
by `finite_value_score`, plus the final finite flags and existing residuals. The
second pass will add per-time stage maxima/minima and finite masks only where the
first pass localizes the failure.

## Skeptical audit

| Risk | Resolution |
|---|---|
| Mistaking a new claim for diagnosis | Consumed seeds are explicitly diagnostic-only and no mean/CI is produced. |
| Changing controls while diagnosing | Exact selected controls are frozen for reproduction. |
| Broad tracing changing numerics | Start with existing outputs; add scalar stage summaries only after localization. |
| Missing structural mismatch | Pre-reset structural residual remains a hard diagnostic veto. |
| CPU/GPU mismatch | Trusted GPU, FP32/TF32 and XLA are required. |
| Unsupported root-cause claim | Report `confirmed`, `ruled out`, and `not checked` separately. |

Audit verdict: `PASS_FOR_BOUNDED_DIAGNOSTIC_EXECUTION`.

## Budget and artifacts

- One eight-seed increment reproduction.
- One targeted stage-trace replay for the failing seed, plus one focused parity
  check if required.
- Maximum 10 minutes trusted GPU time.
- Artifact root:
  `docs/benchmarks/artifacts/genut_str_ukf_nonfinite_root_cause_20260722/`.
- Result note:
  `docs/plans/bayesfilter-genut-str-ukf-nonfinite-root-cause-result-2026-07-22.md`.
