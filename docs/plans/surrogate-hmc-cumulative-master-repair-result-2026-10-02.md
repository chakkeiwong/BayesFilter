# Cumulative surrogate-HMC master repair: execution and review

The repair is complete. The goal remains testing surrogate HMC on the frozen
M13 importance-likelihood posterior. Every fixed-kernel post-warmup draw is
counted as a posterior sample. The public tuner's historical verification label
does not exclude those draws, and a passing cumulative R-hat check does not
start an independent validation chain.

The shared controller is bayesfilter/testing/surrogate_hmc_m13_cumulative.py.
The deadline master, resume master, and recovery master's retained_stage all
call it. The older standalone retained wrapper reaches that same stage. The
suffix-only controller is explicitly historical and is exercised only by its
comparison fixtures. The public tune_hmc_kernel retains artifact authority;
the numerical target, force, affine transform, and frozen runner are unchanged.

## Review findings and completed repairs

- Count the tuner's eligible prefix and every saved continuation once. Exclude
  warmup and assess the complete saved history before constructing a runner.
- Stop at cumulative R-hat <=1.01 with numerical health, or at the authorized
  deadline/20,000 total draws per chain. Check terminal partial intervals too.
  ESS, MCSE and acceptance are reported; low ESS does not create another phase.
- Reject checksum, binding, seed, state, step-size and health inconsistencies.
  Numerical dependencies must match the original run; only the four reviewed
  orchestration files have a checked source migration. Saved snapshots remain
  required. The extra standalone wrapper is outside that original source closure.
- Reassess legacy suffix-only qualified results; do not inherit their pass flag.
  Separate sampling qualification from reference equivalence and all-model
  qualification in master results and refreshed next-phase plans.
- Correct the legacy wrapper's obsolete result-status test. An executed but
  scientifically inconclusive assessment is not reported as a worker crash.

The HMC tuning interface and capability registry were inspected. The selected
raw-coordinate typed endpoint-corrected deterministic force remains within
`tune_hmc_kernel`; it is not routed through the fixed-transport tuner. No new
numerical method, retuning, dtype change or GPU sampling was introduced.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Initial focused regressions | 35 passed; 37.50 s wall | artifacts/surrogate-hmc-cumulative-master-repair-20261002-01/tests-attempt-01 |
| Expanded controller/master/legacy tests | 63 passed; 59.16 s wall | artifacts/surrogate-hmc-cumulative-master-repair-20261002-01/tests-attempt-02 |
| Additional standalone wrapper checks | 2 passed; 4.67 s wall | artifacts/surrogate-hmc-cumulative-master-repair-20261002-01/wrapper-tests-attempt-01 |
| Final repaired-master archive execution | Exit 0; 4.47 s wall; zero new transitions | artifacts/surrogate-hmc-cumulative-master-repair-20261002-01/archive-attempt-02.json |
| Independent diagnostic comparison | Prefix and full diagnostics exactly equal; pooled sample bytes identical | artifacts/surrogate-hmc-cumulative-master-repair-20261002-01/final-review.json |

The final 65 distinct regressions include executable consumer wiring, a real
CPU/XLA Gaussian tuner and continuation, no extra runner after a passing prefix,
total-cap accounting, terminal partial checks, corrupt archive rejection,
legacy pass reassessment, source-migration restrictions and wrapper result
propagation. Gaussian fixtures establish engineering behavior only. GPU devices
were intentionally hidden for tests and archive analysis. Two TFP deprecation
warnings were recorded; no tests failed. Full commands, logs, timings and source
snapshots are preserved in the listed directories. No independent reviewer was
invoked; this was source review plus executable regression and archive replay.

The executed archive command was:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_deadline_extension.py reclassify --input docs/plans/artifacts/surrogate-hmc-m13-midnight-20261001-01 --output docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-02
```

## Correct result and refreshed continuation

The [corrected result](artifacts/surrogate-hmc-m13-reclassified-20261002-02/result.md)
reports four chains, with 9,000 warmup transitions per chain excluded.
The first 9,000 post-warmup draws have R-hat 1.00742/1.00547. They passed the
agreed sampling rule, and the master should have stopped at that point.
The old master nevertheless produced 3,750 further draws. Including them gives
12,750 draws per chain and R-hat 1.05418/1.00884. The process-scale parameter
therefore fails the agreed cumulative mixing threshold in the complete archive.
The observation-scale parameter passes. A passing earlier prefix cannot erase
later adverse mixing evidence. All numerical and continuity checks passed.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Repair complete | 65 focused regressions and exact independent archive agreement | No engineering failure found | Long future M13 continuation not rerun | Use repaired cumulative controller | Broad all-model readiness |
| Earlier prefix passed sampling rule | Both R-hat values <=1.01 | Health passed | Later slow exploration | Preserve and correctly label prefix | Guaranteed permanent convergence |
| Complete archive mixing-inconclusive | Process-scale R-hat 1.05418 | No numerical or continuity veto | Process-scale exploration and MCSE | Keep all samples; preserve saved continuation state | Rejection of surrogate HMC as a method |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical, target-status, state, seed or source-integrity failure found |
| Statistically supported ranking | None; no method comparison was performed |
| Descriptive-only differences | ESS, means, MCSE and acceptance are reported without superiority claims |
| Default-readiness | Not established; exact-likelihood equality and all-model success remain unclaimed |
| Next evidence needed | Cumulative mixing if the existing campaign receives another time allocation; no independent validation chain |

The repair and requested reclassification are finished. The prior midnight GPU
allocation has expired, so no new research sampling was launched. There are
7,250 draws per chain left below the owner's 20,000 total cap. Any later
sampling is continuation of this same master, using the same saved chains and
all 12,750 existing draws; its next scheduled diagnostic is at 13,000. It
requires a new time allocation, not a new validation phase or research direction.

Post-run review: later exploration is the strongest explanation for deterioration
of the early diagnostic. R-hat is a diagnostic rather than a proof of permanent
convergence. The repaired stopping policy answers the owner's sampling request;
it does not turn the frozen importance likelihood into the original exact model
likelihood or establish posterior-reference equivalence. The original reports,
source snapshots and both reclassification versions remain intact.
