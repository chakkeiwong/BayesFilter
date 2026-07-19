# SSL-LSTM NeuTra Proper-Score Direct Calibration Result

Date: 2026-07-17

Decision: `CURRENT_4096_8192_CANDIDATE_REJECTED_POWER_REPAIR_REQUIRED`

## Outcome

The controlled run validly rejects the current 4,096/8,192-draw calibration
candidate. It does not reject the dual-loss validation method, NeuTra, HMC, or
the SSL-LSTM. Both prospective rungs completed with zero invalid numerical
replications and zero false decisive results, but the persistent negligible
mean and one-horizon material variance families remained severely
underpowered at 8,192 draws.

The run also revealed that the prospective Monte Carlo coverage-certification
design was underpowered: 256 replications with one-sided tail alpha `0.05/88`
gives a true 95%-coverage procedure only about 26% probability of certifying a
coverage lower bound of 90%. That flaw does not explain the large decision-
power failures, but it prevents interpreting the per-family coverage screen as
a clean covariance rejection.

## Primary Results

| Required family | 4,096 required decision | 8,192 required decision | 8,192 simultaneous lower bound | Interpretation |
| --- | ---: | ---: | ---: | --- |
| IID null | `100%` equivalence | `100%` equivalence | `97.12%` | Required decision passed |
| Dependent null | `8.20%` equivalence | `100%` equivalence | `97.12%` | Larger rung repaired decision power |
| Persistent negligible mean `0.05` | `0%` equivalence | `1.95%` equivalence | `0.26%` | Decisive failure at both rungs |
| Persistent negligible variance `1.05` | `0%` equivalence | `96.48%` equivalence | `91.06%` | Larger rung repaired decision power |
| Persistent material mean `+/-0.20` | `96.09%/97.27%` material | `100%/100%` material | `97.12%/97.12%` | Required decision passed at 8,192 |
| Local material mean `+/-0.20` | `26.17%/23.05%` material | `87.50%/81.64%` material | `79.50%/72.70%` | Improved, but gate not certified |
| Persistent material variance `1.25/0.80` | `94.53%/91.80%` material | `100%/100%` material | `97.12%/97.12%` | Required decision passed at 8,192 |
| Local material variance `1.25` | `0.39%` material | `29.69%` material | `20.89%` | Decisive failure at both rungs |

All 11 required families had zero false decisions and zero invalid procedures
at both rungs. Their simultaneous upper bounds were `2.88%`, below the frozen
5% targets.

## Coverage Audit

No required family passed the per-family simultaneous coverage lower-bound
gate at 4,096; only the local positive mean family passed it at 8,192. That
screen had insufficient Monte Carlo replication capacity. At `n=256`, passing
requires at least 246 covered replications (`>=96.09%` observed), even though
the confidence region is designed for 95% coverage.

Pooled descriptive coverage across the 11 required families was:

| Draws/chain | Covered / total | Estimate | Ordinary pooled 95% exact interval | Role |
| ---: | ---: | ---: | ---: | --- |
| 4,096 | `2558/2816` | `90.84%` | `[89.71%, 91.88%]` | Descriptive only |
| 8,192 | `2606/2816` | `92.54%` | `[91.51%, 93.49%]` | Descriptive only |

Pooling is not a substitute for family-specific coverage because families are
different laws. It does show that the evidence does not support a gross
zero-ridge HAC breakdown. The next design should use at least approximately
1,024 replications per family if it retains the same 88-claim exact-binomial
screen: at true 95% coverage, that gives about 99.8% probability of certifying
the 90% lower target, compared with about 26% at 256.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject 4,096 as the direct design | Multiple equivalence and local-material decision gates failed | No run-validity veto | None material to this rung | Do not use 4,096 for confirmation | The method or target is invalid |
| Reject 8,192 as the complete direct design | Persistent negligible mean and local variance failed badly; local means were not certified | No numerical, false-decision, ridge, GPU, XLA, or artifact veto | Required sample size above 8,192 | Prospective larger-draw preflight and direct repair | 8,192 target G/H values would produce the same outcome |
| Treat per-family coverage decision as underpowered | The planned Monte Carlo screen had only about 26% certification power at true 95% coverage | Evidence-capacity flaw in plan, not covariance veto | Family-specific coverage with more replications | Raise controlled replications prospectively | HAC coverage passed every family |
| Keep HMC and G/H confirmation closed | No controlled candidate passed all gates | Authority and evidence boundary intact | Larger-draw cost and power | New reviewed repair plan | G/H equivalence or material difference |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Passed for execution validity: GPU/XLA, finite generation, zero-ridge HAC, exact loss extrema, KKT checks, strict receipt, and no confirmation access |
| Viable candidates | Neither 4,096 nor 8,192 is viable as the complete controlled design; the dual-loss method remains a viable research direction |
| Statistically supported ranking | None; no method or sampler ranking was attempted |
| Descriptive-only differences | Continuous point losses, condition numbers, runtimes, pooled coverage, and differences between family decision rates |
| Default-readiness | Not established; no API/default or scientific promotion was made |
| Next evidence needed | Prospective larger draw ladder, at least roughly 1,024 controlled replications per family for the retained simultaneous coverage screen, independent seeds, and unchanged scientific thresholds |

## Negative-Result Separation

- Implementation failure: the first smoke exposed unsupported XLA variant
  generation and the second exposed compile-heavy unrolled loops; both were
  repaired before the immutable passing smoke and material run.
- Numerical failure: not supported in the material run; all rows were valid,
  zero-ridge, finite, and KKT-admissible.
- Tuning failure: not tested. `kappa_HAC=1.0` was frozen and must not be tuned on
  these outcomes as if prospective.
- Evidence-design failure: supported for 256-replication simultaneous coverage
  certification.
- Candidate sample-size/power failure: supported for 4,096 and 8,192,
  especially persistent negligible mean and local material variance.
- Evidence against NeuTra, HMC, the SSL-LSTM, or predictive-functional
  validation: not supported.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b1606a3ec19643356705cf9d08ccf7c6495b6186` with unrelated dirty worktree preserved |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TFP `0.25.0` |
| Device | physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`; GPU shared with an unrelated MacroFinance lane |
| Numerical policy | TensorFlow/TFP `float64`; XLA JIT; TF32 enabled; growing Bartlett HAC `kappa=1`; ridge ladder `(0.0,)` |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Seeds | smoke `(27101,27102)`; 4,096 `(27201,27202)`; 8,192 `(27301,27302)` |
| Shape | four chains/arm, two forecast replications, ten horizons, 256 experiment replications/family |
| Wall time | `247.5009` seconds material; 4,096 rung `101.6126`; 8,192 rung `145.8344` |
| Material command | `CUDA_VISIBLE_DEVICES=1 ... run_ssl_lstm_neutra_proper_score_direct_calibration_2026_07_17.py --mode material ... --wall-cap-seconds 9000` |
| Smoke receipt | `proper-score-direct-calibration-smoke.json`, SHA-256 `7554ac456684e02eb802f60320fb7fda927df5d0159bdbfee29402873159398b` |
| Material receipt | `proper-score-direct-calibration-material.json`, SHA-256 `fc4781d98a69fbf1002c0f2b76955e023abde4471187c48a6df01f47e712ebf7` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-proper-score-direct-calibration-plan-2026-07-17.md` |
| Result | This file |

The material receipt records both generator and feature wrappers tracing once
at each draw shape. HMC, retained archives, and G/H confirmation forecasts were
not read or executed.

Receipt source-integrity audit: the current predictive module and runner match
their receipt hashes exactly (`d43eba72...d5b8` and `7d5c0490...2fe`). The
receipt-bound pre-run plan hash is `51b23d14...8bf5`; the current plan hash is
`cd98299a...646c` because its status and material-close section were appended
after the immutable receipt was written. No pre-run numerical contract was
changed. The receipt was not rewritten.

## Verification

| Check | Result |
| --- | --- |
| Predictive and direct-calibration focused suite | `95 passed` in `39.25 s`; two dependency deprecation warnings only |
| Post-smoke harness suite | `12 passed`; exact smoke identity and claim boundary checked |
| Python compilation | Passed for the predictive module, runner, and focused tests |
| Scoped diff whitespace | Passed |
| Trusted GPU/XLA smoke | Passed in `14.8377 s`; wrapper traces once; zero invalid rows |
| Trusted material ladder | Completed both rungs in `247.5009 s`; all four compiled wrapper/shape surfaces traced once |
| LaTeX build | `docs/main.pdf`, 405 pages, 1,596,240 bytes |
| New Chapter 28a labels/references | Resolved for horizon loss, dual threshold, and dual decision |

The complete book still reports 11 undefined citations and four multiply
defined labels in unrelated chapters. They predate this lane and were not used
by the new Chapter 28a material.

## Post-Run Red Team

The strongest alternative explanation for the decision failures is not a
scientific defect but conservative 20-dimensional geometry: every average and
horizon loss is bounded over the same `chi2_20(0.95)` region. This provides
joint false-decision control but shrinks slowly near the declared anchors. The
anchor clearance is particularly small for persistent negligible mean and
local variance.

The conclusion that 8,192 is underpowered would be overturned by a discovered
truth-sign, covariance-scaling, trust-region, or seed-independence error. Those
surfaces have focused scalar/batched tests and zero-invalid material evidence,
but independent re-review remains appropriate before a larger run. The weakest
part of this result is the per-family coverage certification because its
replication count was underpowered by design.

## Handoff

Write a new repair plan before any additional serious run. It should:

1. preserve `K_avg=K_max=0.0068491`, equal horizon weights, `kappa_HAC=1.0`,
   zero ridge, and the same required families;
2. use a prospective projection or analytical clearance calculation to choose
   a larger draw ladder, with direct validation rather than projection as the
   promotion evidence;
3. use at least approximately 1,024 replications per required family if the
   `0.05/88` simultaneous coverage contract is retained, or prospectively
   justify a less conservative exact simultaneous construction;
4. use new independent seed domains and a fresh resource cap;
5. keep HMC and G/H confirmation closed until a direct controlled candidate
   passes.
