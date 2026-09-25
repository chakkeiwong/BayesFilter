# Control-safety result and next implementation phase

The 108-row screen completes after one validator repair. It exposes substantial distortion from the inherited coordinate cap without establishing a replacement setting. Every tested LEDH setting has higher observed score error than at least one constructed cheap comparator in both weak and concentrated observation regimes. Those observations veto promotion of these configurations; they do not reject LEDH, covariance alternatives, or the planned repairs. Two datasets and one particle stream per condition do not support a statistical ranking.

## Evidence and computed target

The [plan](../../../younis-score-control-safety-calibration-2026-09-16.md) fixes N=128, T=4, three scalar nonlinear regimes, datasets 700/701, seed 92167, 14 LEDH controls and four cheap comparators. The target is the model marginal-likelihood score. Each particle row computes the analytical derivative of its declared finite filtering program; that derivative differs from the model score. Error is measured against an independently refined FP64 numerical grid, with its refinement and tail checks. No unbiased-model-score claim follows.

The full [conditional table](control-safety-screen-02/conditional-diagnostics.md) and [machine-readable report](control-safety-screen-02/conditional-diagnostics.json) retain every dataset, score vector, oracle, diagnostic range, result path, source revision and runtime record. Reporting checks all 108 result checksums, exact paired data/reference signatures, and 36 successful rows per condition. All rows use GPU, FP32, TF32 and XLA with verified pre-initialization memory growth. Concurrent host work excludes timing comparisons.

| Regime | Baseline observed score MSE | Local observation proposal MSE | LEDH control screen |
|---|---:|---:|---|
| Weak | 4.48537 | 0.184644 | Every control fails observed heuristic screen |
| Curved | 0.488030 | 0.511441 | Several controls pass this descriptive screen; no ranking |
| Concentrated | 5.77141 | 1.06098 | Every control fails observed heuristic screen |

The local observation proposal is one of four predeclared falsification comparators, alongside bootstrap PF, EKF and UKF. Their errors do not select safety controls. The curved-regime result differs from the preceding pilot on other datasets, illustrating why these two-path comparisons cannot establish superiority.

## What the diagnostics show

At coordinate cap 0.4, mean intermediate standardized-coordinate displacement ranges from 0.451 to 0.501 across the three regimes, and the minimum cap derivative reaches 1.71e-9. At cap 8, displacement ranges from 3.53e-7 to 9.27e-6 and the minimum derivative is at least 0.99728. The coordinate map is therefore materially compressing ordinary baseline clouds. Its later affine moment restoration may expand them again: these measurements neither bound final particles nor prove that this mechanism causes the model-score error.

Caps 4 and 8 reduce intermediate displacement but do not rescue the observed score errors. Doubling flow substeps also fails to give uniform error reduction. Increasing Sinkhorn/balance from 30/30 to 60/60 changes score vectors by at most 1.37e-6 in these fixtures. The baseline correction-system condition diagnostic ranges up to 67.25; a universal condition-number cutoff was neither used nor inferred. Scalar fixtures have inactive pairwise corrections and cannot calibrate their controls.

The cap has form `g_c(z)=z/(1+(z/c)^p)^(1/p)` for finite positive c and even p. Its derivative is `(1+(z/c)^p)^(-1/p-1)`. Thus its intermediate bound needs c>0, not c<1. MathDevMCP verified the local derivative simplification in [control-cap-mathdev.json](control-cap-mathdev.json); two parser failures are retained. This is a checked local identity, not certification of the complete algorithm.

## Implementation, failures and verification

The shared canonical caller now retains the correction diagnostics it previously discarded. Opt-in Gaussian and nonlinear consumer traces expose them without copying the correction algorithm. Host guards reject nonfinite diagnostics and invalid covariance or correction status; they do not alter accepted results. Actual call-chain tests cover all three covariance consumers and a two-state pairwise route.

The initial GPU smoke completed 8/8 rows. Nonlinear value/score parity was exact; the Gaussian maximum discrepancy was 3.5763e-6, within the predeclared FP32 envelope. The first screen completed 32 rows and rejected four cap-4/8 rows during tracing. Inspection found that the generic standardized cap inherited the bounded-chart upper limit. The repair changes only pre-trace validation, rejects nonfinite caps, and retains the separate bounded-chart restriction. [The AST audit](control-safety-repair-config-01/repair-audit.json) checks that accepted numerical operations are unchanged. The final launch executes only four rejected and 72 previously unstarted rows. Preserve the failed attempts; they are implementation failures, not negative evidence against these cap values.

Frozen revisions are `41446e1e38c8fad4f837852bf54d575f18c535f9` and repaired `cbcfea1eab939f803ea84215cb7dbb2bcb090704`. Seven diagnostics tests, two canonical regressions, thirteen cap/moment/analytical-derivative tests and six mandatory commit oracle checks pass. Early import-order and test-argument failures are preserved and repaired. Integration into the shared checkout verifies all nine predecessor files before copying; [the integration audit](control-diagnostics-main-integration.json) records before/after hashes. The seven actual main-checkout consumer tests also pass (`control-main-consumer-tests-01.log`). No unrelated working-tree changes were reverted.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept diagnostic implementation | Actual consumer wiring and parity pass | No unresolved engineering veto | Small tested scopes | Retain diagnostics in subsequent calibration | General numerical validity |
| Accept cap validator repair | Domain derivation and unchanged accepted arithmetic checked | Focused tests and resumed rows pass | Larger caps on other models | Permit explicit calibration candidates | Default cap selection |
| Withhold all method/default promotions | No promotion criterion in this screen | Weak/concentrated observed heuristic vetoes | Two paths; finite-program and reference error | Fresh scope-specific calibration and replication | Statistical ranking or direction rejection |
| Continue master implementation | This screen answers its bounded diagnostic question | No continuation veto remains | Adaptive-N iAPF reporting still unavailable | Implement frozen realized-N selection and accounting | Complete master execution |

| Inference status | Finding |
|---|---|
| Hard veto screen | No unresolved nonfinite/reset/reference/provenance failure in the 108 completed rows; four repaired validation failures remain recorded |
| Statistically supported ranking | None |
| Descriptive-only differences | All MSE, cap displacement, derivative, transport sensitivity and between-control differences |
| Default-readiness | Not established; no default changed |
| Next evidence needed | Fresh calibration of the complete relevant control family, frozen selection, independent replicated model-score comparisons and multidimensional coverage |

Strongest alternative explanation: the model-score error may primarily arise from finite-cloud/reset/flow approximation rather than coordinate compression; the larger-cap rows do not rescue it. More datasets could also overturn observed ordering. The weakest evidence is the two-dataset scalar coverage. An intermediate residual or covariance improvement cannot substitute for downstream score evidence.

## Budget and phase refresh

GPU wall time, including failures and compilation, is 102.324 + 151.523 + 395.948 = **649.795 seconds of 2,700**. All three launch slots and 120 numerical attempts are used (8 smoke + 36 first screen + 76 repair). This allocation is closed; unused time is not authority to repeat its exhausted attempts. Timed CPU command wall totals 460.73 seconds of 3,600, including failed tests and commit hooks; raw `.time` files retain CPU user/system usage. Pytest durations and process timer durations are separate measurements. Reporting itself is a short standard-library post-run check.

The next master implementation obligation is [adaptive-N iAPF selection/reporting](../../../younis-score-iapf-adaptive-scope-2026-09-16.md). Its design must distinguish the requested starting count, realized offline count, fitting cost and final fixed-count evaluation. Hyperparameters are selected before heldout evaluation; the selected procedure may fit to new observations using independent offline streams, then freeze coefficients and count before its final sample. This corrects an overstrict initial refresh that would have forbidden the per-dataset fitting already defined by iAPF. Scalar Gaussian mechanics remain the initial bounded scope, with Kalman and untwisted/one-step/fitted proposal comparators. Nonlinear iAPF, complete LEDH numerical-control calibration, broader horizons/counts, normalization/consistency calibration, combinations, matched-cost replication and terminal review remain explicit subsequent work. Opened datasets 400–423 and this screen's 700/701 remain excluded from fresh claim data.
