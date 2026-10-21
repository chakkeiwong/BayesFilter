# Recursive precision localization within the HMC repair campaign

Question: can FP64 storage and small-matrix recurrence, combined with centered
FP32 particle-pair kernels, recover the SIR score without paying for dense FP64
pair work? This is an explicitly mixed-precision diagnostic. FP32 remains the
campaign default. It does not issue a tuning artifact or enter HMC.

The direct FP32/FP64 SIR T50 pair failed the score screen (maximum scaled gap
0.48305) although the likelihood absolute gap was 0.001147. Both rows were
healthy. Changing the Gaussian contraction did not close the gap. The next
discriminating experiment separates recursive-state precision from the expensive
pairwise arithmetic. Do not interpret this as proof of a unique root cause.

Use the same saved data, N4032/K2016, B2, parameters, direct contraction,
baseline controls and FP32-quantized initial inputs as that pair. The first run
uses its first ten observations; compare with sums of the first ten saved FP64
likelihood and score increments. Verify parameter/input provenance, controls,
beta, data checksum and numerical source hashes. Stored prefix increments are
a comparator only after those checks pass. TF32 stays off and XLA stays on.

Load the unchanged shared numerical module into a private diagnostic namespace;
replace only the Gaussian-mixture and transport entry points with typed
wrappers. The wrappers call the existing kernels. No reduced numerical
implementation, global monkeypatch, new parameter derivative or new reset is
introduced. Save the wrapper and shared-source hashes in every worker snapshot.
All outer recurrence, local model callbacks, moments and recolouring remain
FP64. Pairwise kernels use FP32 after centering in FP64. This precision pattern
is a hypothesis, never label it pure FP32.

Translation argument: Gaussian mixtures depend on x_i-m_j, so replacing both by
x_i-c and m_j-c leaves densities unchanged. Tangents become dx_i-dc and dm_j-dc;
their difference is unchanged. Transport costs depend on x_i-x_j. For normalized
column weights, y_j=sum_i w_ij x_i=c+sum_i w_ij(x_i-c).
Differentiate both terms, including dc; this is the same total derivative in
real arithmetic. Centering before a lower-precision cast avoids throwing away
small differences between large absolute positions. It does not prove the
remaining FP32 reductions accurate.

Primary localization criterion: healthy complete rows and every cumulative
value/score component within 1% of the matching FP64 prefix, using
max(1,abs(reference component)). Numerical invalidity vetoes that arm.
Parameter/data/source mismatch vetoes the experiment. Runtime, stepwise gaps
and tangent norms explain the result and supply no statistical ranking.
A prefix pass triggers the same mixed policy at T50; it never promotes HMC.
A failure permits Gaussian-only and transport-only FP32 isolation. A healthy
T50 pass requires fresh own-scope calibration/validation before any HMC use.
Default-readiness, exact-model likelihood accuracy and posterior convergence
are explicitly unestablished.

CPU checks first: ensure isolation leaves the runtime module unchanged; centered
wrappers agree with independent FP64 kernels on offset fixtures and preserve
validity; exercise the shared full filter through stable CPU XLA. CPU is a
diagnostic exception with GPU hidden. GPU budget: at most 2400 worker seconds
total for prefix/full-horizon/isolation, charged to the existing ten-hour ledger.
At most four attempts, no concurrent GPU job. Stop if source or input provenance
cannot be reconciled. Preserve every failure. Output roots use
surrogate-hmc-sir-precision-islands-20261012-*.

Assumption audit: FP64 recurrence is a proposed precision island, justified by
the observed early recursive divergence; it may still fail when dense FP32
roundoff is amplified. Saved inputs are localization data, not fresh validation.
Translation identities justify centering; the cheap offset tests check their
implementation. The 1% threshold and N4032 are owner choices. Baseline controls
are frozen comparators, not newly tuned defaults. This arithmetic ablation has
no method-ranking claim; practitioner baselines remain necessary for the later
model-level comparison.

Skeptical audit: avoid confusing mixed arithmetic with FP32; preserve all tangent
and validity outputs; avoid rerunning a changed data prefix; do not let a prefix
pass qualify the full horizon. The proposed wrappers share the original
call chain and cannot leak into live workers. Proceed after CPU checks and the
active LG force-reuse experiment.

Prelaunch audit completed 2026-10-12: four CPU/XLA precision-wrapper tests pass;
65 launcher/reporting checks pass. Every saved FP64 source/dependency hash still
matches; both reference rows completed T50 with valid status. Prefix comparison
rejects source/data/parameter/control mismatch and checks that saved increments
reconstruct the reference total. First GPU command:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py run --model sir_d18 --output docs/plans/artifacts/surrogate-hmc-sir-precision-islands-20261012-01 --gpu 0 --batch-size 2 --paired-repeats 0 --offset 0.0 --budget-seconds 600 --per-model-timeout 600 --campaign-ledger docs/plans/artifacts/surrogate-hmc-repair-continuation-20261012-01/budget.json --score-only --fp64-reference --debug-horizon 10 --precision-islands both_fp32 --shared-gaussian-contraction direct
```
