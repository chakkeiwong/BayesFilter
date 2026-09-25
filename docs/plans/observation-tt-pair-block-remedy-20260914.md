# Pair-block SGQF/TT remedy: review and bounded test plan

## Question and target

Can the diagnosed first-transition regression error be reduced by representing
the adjacent state jointly in paired blocks `(u_i,v_i)`, while retaining the
SGQF observation guide, exact retained marginal, particle-specific conditional
proposal, converged fitting diagnostics, and importance-weighted regression
rows? Here `u` is the current state in its SGQF chart and `v` is the previous
state in the previous posterior chart. The pair-block path is an
`extension_or_invention` relative to Zhao--Cui; it is not claimed to be their
TT-cross route.

## Evidence contract

The exact target is the same adjacent-state square-root amplitude used by the
existing driver. The comparator ladder is: (i) existing grouped scalar TT,
(ii) scalar paired-order TT-SVD reference, and (iii) pair-block TT with the
same degree, rank budget and target rows. The primary representation screen is
finite weighted relative RMS on a fresh calibration split, followed by an
untouched audit split. The downstream screen is agreement of filtering means
and log evidence with the validated scalar-grid/reference ladder on fresh
observations. A proposal/CDF/Jacobian mismatch, non-finite value, invalid
positive mass, or failed independent normalization is a hard veto. Fit loss,
ESS, and runtime are explanatory diagnostics and repair triggers, not proof of
filtering quality. No ranking is claimed from four seeds or one sequence.

The cheap heuristic adversaries are constructed from the model: transition
bootstrap (the exact conditional dynamics), stationary-prior proposal (a
state-independent prior), and the SGQF Gaussian guide (the moment-matched
observation proposal). Filtering is evaluated separately for near-zero,
ordinary, and large observations. Losing to one of these in a salient regime
vetoes promotion while leaving the repair experiment viable.

## Assumption and default audit

The paired order is motivated by the measured near-diagonal transition and the
rank-three middle-cut obstruction. Degree three and rank three are the frozen
baseline inherited from the diagnosis; ranks two and four are a bounded
sensitivity ladder, not promoted defaults. The initial L1 grid is
`{0,10^-5,10^-3}` because it is the current scope's grid; selection uses only
fresh calibration validation rows. The fitter records objective decrease,
effective rank, and KKT residual; four sweeps and 128 proximal steps remain a
baseline and may trigger a bounded repair to eight sweeps and 256 steps. Rows
are drawn from `s=eps*rho+(1-eps)*s_joint`, eps=0.2, with weights rho/s <= 5;
the mixture is frozen before validation and audit. These choices can fail
through poor rank, gauge-dependent L1 shrinkage, row-weight variance, or
non-converged ALS; the smallest exposing diagnostics are the decomposition
singular values, weighted/unweighted objective, ESS of row weights, and KKT
residual respectively.

## Implementation and tests

1. Add pair-block cores `C_i[a,alpha,beta,c]`, batch-safe evaluation,
   conversion to the existing scalar KR authority for independent-reference
   checks, exact integration of past coordinates as positive matrix maps, and
   contraction of past coordinates into particle-specific current scalar cores.
2. Add weighted Hermite regression with train/validation/audit separation,
   frozen log-scale, bounded convergence diagnostics and rank reporting. Keep
   the unweighted fitter as the exact comparator.
3. Add the Gaussian backward conditional guide from SGQF moments and a frozen
   defensive random-mixture row sampler. Verify `E_s[(rho/s)(h-H)^2]` against the
   direct Gaussian integral and record the weight bound and effective sample
   size.
4. Add focused CPU reference tests for pair evaluation, marginal contraction,
   conditional normalization and particle dependence; serialize/reload cores
   and charts; verify physical-space log densities and Jacobians. Run the
   existing observation-guided suite unchanged.
5. Execute a fresh representation ladder on dimensions 2 and 4 with a new
   calibration/validation/audit seed partition. If the baseline fails its
   convergence diagnostic, perform at most one predeclared repair (eight
   sweeps/256 steps) within the remaining campaign budget.
6. Execute at most one fresh downstream d=1 and d=4, T=20 filtering launch
   with four seeds and the heuristic ladder. Preserve all output in a new
   versioned directory; never tune on its audit data.

## Skeptical pre-run review and stops

The pair block changes the representation, so lower residual alone cannot
establish a better filter. The exact marginal and conditional consumers are
therefore mandatory. A scalar paired-order reference is a diagnostic authority,
not a production implementation. The importance proposal changes only the
sampling measure and its explicitly included rho/s weight; an unweighted or
randomly renormalized objective is a veto. Stop the campaign on corrupted
fixtures, invalid reference normalization, non-finite or negative masses,
missing audit separation, or the two-launch/remaining-time budget. A failed
candidate is evidence for the predeclared solver/rank repair; it does not reject
the pair-block direction.

## Environment, budget and artifacts

Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, TensorFlow float64,
GPU/XLA with memory growth for the candidate launch, and explicit
`CUDA_VISIBLE_DEVICES=-1` CPU/non-XLA only for the focused reference tests.
The inherited campaign has about 2,275 numerical seconds and two full launches
remaining. Every launch writes `run_manifest.json`, `result.json`, complete
logs, git commit, environment, seeds, hardware and plan path under
`docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/`.

## Decision record

The implementation is accepted as an executable extension only if all focused
identity/normalization tests pass. Filtering promotion additionally requires
the predeclared reference screens and no heuristic-dominance veto. Even then,
four seeds and one sequence do not establish statistical superiority, broad
scaling, posterior correctness, HMC readiness, or source-faithful Zhao--Cui
TT-cross behavior.

## Concrete launch specification and review, 2026-09-14

Review verdict: executable as a bounded diagnostic and one downstream test.
The review caught and repaired (before the launch) confusion between backward
conditional and smoothed marginal covariance, grouped-to-paired row indexing,
zero-initialized inactive rank channels, duplicated KR contraction logic, and
random-sum weight normalization. Eighteen focused CPU tests pass, including the
shared unbatched sampler; compilation/import checks pass. No independent model
review is claimed. MathDevMCP theorem extraction is advisory and inconclusive
where its algebraic backend cannot encode the theorem; explicit identities are
audited separately.

Fresh partitions: model parameters are copied from the frozen C2 fixture, but
observations are simulated by TensorFlow with seeds 2026091401 (calibration)
and 2026091402 (downstream). Dimension two uses the leading transition submatrix
and recomputes its stationary covariance. Training/validation/audit row seeds
are separated by the fitter; an additional common Gaussian audit uses 58100+d.
No audit residual selects a configuration. The diagnostic target uses the SGQF
Gaussian incoming law at t=1 to isolate representation and fitting; it is not
the recursive TT target. The downstream master tests the latter.

The bounded diagnostic compares scalar grouped and scalar interleaved fits
using the same scalar fitter, and pair-block fits at ranks 2,3,4 with and without
importance rows. This replaces a new expensive TT-SVD projection: the existing
TT-SVD evidence already established representation capacity, whereas this test
asks whether the finite fitter realizes it. Pair/scalar initialization differs
and must be reported as a confound. Degree three, 1024 rows per split, and the
L1 grid above are frozen. KKT > 0.001 in the rank-three weighted baseline
triggers the single eight-sweep/256-step solver repair. This threshold is a
scale-dependent explanatory diagnostic on a training-normalized target, not a
convergence certificate or promotion criterion. Select the weighted pair rank
and solver on d4 validation only. Apply that configuration to d1 and d4 in the
fresh downstream check; this is a transfer diagnostic, not d1-specific tuning.
The downstream seeds are 2101--2104, training seed 64100 and reference seed74100.

Budget: at most 650 seconds for the representation diagnostic, then at most
1500 seconds for the one downstream launch, within the inherited approximately
2275 seconds. A localized harness failure can use the unconsumed allowance;
all attempts and wall times remain recorded. Check budget between stages and
use an external timeout for a hard process limit. Invalid algebra, nonfinite
results, invalid proposal CDF brackets, missing split provenance, or corrupted
fixtures are continuation vetoes; high residual/KKT or heuristic losses reject
promotion and may trigger only the already specified repair.

Commands (environment: tftwogpu; set CUDA_VISIBLE_DEVICES to RTX5080 UUID,
TF_FORCE_GPU_ALLOW_GROWTH=true, TF_NUM_INTRAOP_THREADS=2,
TF_NUM_INTEROP_THREADS=1; escalated execution):

    timeout 650s python docs/benchmarks/diagnose_pair_block_tt_remedy.py --output-root docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/diagnostic-01 --fixture-source docs/benchmarks/artifacts/observation_aware_tt_complete_20260913/c2_fixture.json --wall-budget-seconds 640

The full-master command consumes diagnostic-01/downstream_fixture.json and
selected_configuration.json (rank and pair solver settings), writes campaign-01,
and sets --pair-block --dimensions 1 4 --horizon 20 --particles 512
--reference-particles 32768 --seeds 2101 2102 2103 2104 --training-seed 64100
--reference-seed 74100 --wall-budget-seconds 1490, under timeout 1500s.
Its run manifest preserves the exact expanded command. The naive transition,
stationary-prior, SGQF Gaussian, scalar predictive/guided, and pair proposals all
use the same observations and particle counts. No scalar baseline is silently
changed to the pair rank: retain its frozen rank three when the selected pair
rank differs. Preserve all heuristic regimes actually observed; absent regimes
are missing evidence, not passes.

Diagnostic-01 stopped after 6.188 seconds: the harness called fit_from_log instead of the existing fit_amplitude API. No regression ran. CPU/GPU compiled conditional parity passed (maximum gap 2.22e-16). The local call-name repair preserves target, data, seeds and method; diagnostic-02 uses the same generated fixtures with a fresh output directory and at most640 seconds. This is a harness failure, not candidate evidence.
