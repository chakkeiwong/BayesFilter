# SSL-LSTM q=20 seed-B five-horizon energy diagnostic plan (2026-08-09)

## Objective

Compare two fixed black-box simulators: the q=20 SSL-LSTM output simulator at
the authenticated seed-B posterior-mean physical parameter and the same
simulator at the synthetic true-control parameter. At each horizon
`T in (10, 20, 30, 50, 100)`, draw 1,000 independent complete paths from each
black box and perform one whole-path two-sample equality diagnostic.

There are five diagnostics, not four. They are reported separately. No joint
test, combined p-value, familywise pass/fail gate, or predictive-equivalence
claim is computed.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | At each fixed T, does a whole-path two-sample test distinguish the posterior-mean and true-control simulator distributions? |
| Candidate | Fixed posterior-mean vector `(0.6442915353159974, 0.16321747278874743, 0.6098970783640596, 0.1670831614630769)`. |
| Comparator | Fixed true-control vector `(0.35, -0.08, 0.65, 0.05)`. |
| Null at T | The two black boxes generate the same probability law on `R^T`. |
| Alternative at T | Their probability laws on `R^T` differ. |
| Statistic | Biased empirical energy distance on the complete raw T-vector paths. |
| Decision | Reject the T-specific equality null when the balanced-label permutation p-value is `<0.01`; otherwise report `NOT_DISTINGUISHED`. |
| Hard veto | Invalid target/parameter provenance, nonfinite path, failed forecast status, wrong shape, nonfinite/negative-beyond-roundoff energy statistic, invalid permutation geometry, missing XLA provenance, GPU use in the CPU exception, cap violation, or corrupt artifact. |
| Explanatory diagnostics | Energy statistic, permutation distribution summary, mean/variance differences, runtime, and horizon. |
| Must not conclude | Mathematical equality, practical equivalence, posterior correctness, parameter identification, model adequacy, all-horizon DGP equality, NeuTra superiority, or default readiness. |

## Evidence contract

1. The target signature is
   `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
2. The posterior-mean vector and target/adapter identity come from
   `docs/plans/artifacts/ssl-lstm-q20-seed-b-plugin-predictive-comparison-2026-08-08/r4/material.json`,
   SHA-256
   `72ba9c7034e36f26e76d0d6542c3aa0ab6699e4d21fe0f727ca5dea275663f09`.
   Only its authenticated parameter vector is reused; its paths and descriptive
   result are not reused.
3. Every T uses two independent stateless simulator seed banks and an independent
   permutation seed. No path is reused across horizons or arms.
4. Each arm has exactly 1,000 independent complete paths. A path is the unit of
   observation. No within-path time point is treated as an independent sample.
5. Paths remain in their shared raw output coordinate system. No arm-specific
   centering, scaling, whitening, or fitted transformation may erase a real law
   difference.
6. The statistic is
   `E = 2 mean(d(X_i,Y_j)) - mean(d(X_i,X_j)) - mean(d(Y_i,Y_j))`,
   including zero diagonals in the two within-arm V-statistic terms. Euclidean
   distance is computed on complete T-vectors. Under equal sample sizes, the
   same statistic for a balanced label vector `s in {-1,+1}^{2000}` is
   `-s' D s / n^2`, where `D` is the pooled distance matrix.
7. The permutation p-value is `(1 + count(E_perm >= E_observed)) / 10000` from
   9,999 independently generated balanced random label permutations. This is a
   Monte Carlo permutation test with minimum attainable p-value `0.0001`.
8. The per-horizon significance level is `0.01`, user chosen. Equality is
   rejected only for `p < 0.01`; equality at the boundary is not rejected.
9. No multiplicity adjustment is applied because these are explicitly separate
   diagnostics. If five valid tests were independent and all five nulls held,
   the chance of at least one false rejection would be
   `1 - 0.99^5 = 0.0490099501`. This is a familywise error probability, not a
   combined p-value. With disjoint simulation/permutation seeds the Monte Carlo
   test outcomes are independent conditional on the two fixed black boxes, but
   the five scientific hypotheses are related across horizons; no joint
   interpretation is made.
10. The forecast and energy/permutation kernels use TensorFlow XLA. NumPy is not
    used. CPU-only execution is a reviewed diagnostic exception because both
    GPUs were 95-98% utilized by concurrent work at planning time. GPU devices
    are hidden before TensorFlow import. This run supplies no performance,
    production-target, or default-readiness evidence.

## Numeric provenance and defaults

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| `n=1000` per arm | User-fixed diagnostic size | Low power for subtle high-dimensional differences | Null canary plus p-value/non-rejection nonclaim |
| Horizons `10,20,30,50,100` | User-fixed diagnostic grid | Does not cover other finite horizons or the infinite process law | Report T-specific results only |
| `alpha=0.01` | User-fixed per-test level | Readers misinterpret five passes/rejections jointly | No combined decision; report `1-0.99^5` only as independence arithmetic |
| 9,999 permutations | Derived resolution choice | Monte Carlo noise near 1% | Resolution `0.0001`; record exceedance count and exact seed |
| Biased energy V-statistic | Standard finite-sample whole-path equality statistic; reviewed implementation choice | Quadratic memory/cost; sensitivity degrades with dimension | `n=32` null canary, direct-versus-label identity test, shape/finite checks |
| Raw shared coordinates | Equality-target requirement | A dominant coordinate scale can dominate distance | Same scalar observation units at every time point; per-time summaries explanatory |
| CPU/XLA | Reviewed diagnostic exception | Slower runtime or backend mismatch | Canary timing; 7,200-second campaign cap |

## Skeptical pre-execution audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | No. Both arms use the same q=20 simulator and differ only in the fixed parameter vector requested by the user. |
| Proxy promoted? | No. The complete T-vector is tested. Descriptive mean/variance rows cannot change the p-value decision. |
| Missing stop condition? | No. Canary vetoes invalid mechanics. Each horizon writes a unique receipt. Any hard veto stops later horizons; statistical rejection does not stop the diagnostic grid. Campaign cap is 7,200 seconds. |
| Unfair comparison? | No. Equal n, dtype, simulator, backend, and horizon; independent seeds prevent artificial pairing. |
| Hidden assumptions? | Exposed: iid paths, fixed parameters, raw Euclidean geometry, finite T, Monte Carlo permutations, and diagnostic-only multiplicity handling. |
| Stale context? | The failed MMD-equivalence campaign is not a baseline and supplies no threshold. Only the authenticated posterior-mean vector is reused. |
| Environment mismatch? | The simulator currently hard-codes T=10. It must receive a default-preserving XLA horizon parameter and focused replay/shape tests before the campaign. CPU is an explicit diagnostic exception. |
| Could success mislead? | Yes. Five non-rejections can reflect low power and do not prove equality. This is stated in every summary and result artifact. |
| Could rejection be an implementation artifact? | The null canary, direct/stateless replay tests, exact balanced-label identity, finite checks, and independent seeds distinguish mechanics failure from a candidate difference. |
| Would artifacts answer the question? | Yes, narrowly: one statistic, permutation distribution, and p-value for each complete-path law at each fixed T. |

Audit verdict: **PASS AFTER REQUIRED HORIZON-API AND MECHANICS TESTS**. The
serious run may begin after those focused checks and a null true-versus-true
canary pass without hard veto. The canary p-value is mechanics evidence only and
does not calibrate or guarantee Type-I behavior from one realization.

## Execution

Artifact root:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-five-horizon-energy-diagnostic-2026-08-09/r1/`

Phases:

1. Focused unit/identity/XLA checks.
2. Null true-versus-true canary at `n=32`, `T=20`, 999 permutations; cap 900 s.
3. Five diagnostics at `n=1000`, 9,999 permutations, with per-horizon receipts;
   total campaign cap 7,200 s.
4. Terminal result and reset memo.

## Interpretation

- `p < 0.01`: `DISTINGUISHED_AT_1_PERCENT` for that T.
- `p >= 0.01`: `NOT_DISTINGUISHED_AT_1_PERCENT` for that T.
- Neither status is an equivalence conclusion.
- A rejection at one T does not require stopping the remaining diagnostics.
- Passing all five means only that none of the five tests rejected. It is not a
  p-value and does not establish the same DGP.

