# HNN-NeuTra Exact-Gradient Comparison Reset Memo

Superseded reset state, 2026-07-18.  Resume from
`bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.  The
performance state below used non-native tuning and is
`UNSUPPORTED_PENDING_NATIVE_RETUNING`.

Program status: `COMPLETE_WITH_STR_EXACT_COMPARATOR_UNRESOLVED`.

## Established State

- The primary comparison is HNN-NeuTra-HMC versus exact-gradient NeuTra-HMC on
  the same frozen chart. Zero residual is an explanatory ablation only.
- LGSSM-KF, PP-UKF, PP-SGQF, and SIR-SGQF have complete one-seed two-arm
  accuracy evidence and descriptive performance-screen passes.
- STR-UKF HNN passed convergence, truth, and deterministic/no-noise gates.
- STR exact-gradient NeuTra-HMC failed long-warm-up energy health twice, at
  `epsilon=0.2` and `0.1` with `L=8`; no retained exact draws or direct
  posterior comparison exist.
- A healthy STR matched-mechanics benchmark at `epsilon=0.1`, `L=8` observed a
  descriptive `14.031x` exact/HNN time ratio. It does not close posterior
  agreement.
- Corrected nonlinear matched ratios are `25.435x`, `25.895x`, `25.564x`, and
  `14.031x` for PP-UKF, PP-SGQF, SIR-SGQF, and STR-UKF mechanics respectively.
- No statistically supported ranking or default-readiness claim exists.

Terminal result:
`docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-terminal-result-2026-07-18.md`.

## Repairs Preserved

1. Added the missing exact-gradient nonlinear arms and fair same-chart matched
   mechanics benchmark.
2. Synchronized supervision, training, tuning, sampling, and benchmark timing.
3. Required matched-mechanics health for performance promotion.
4. Separated HNN preparation break-even from full independently tuned
   reuse-campaign break-even.
5. Preserved STR attempt 01 and the exact-only attempt 02 repair; the repeated
   failure closes further retries under this runbook.

## Re-Entry

Only one scientific gap remains in scope: STR direct HNN/exact posterior
agreement. Re-enter it as a new, bounded structural exact-HMC stabilization
campaign. Start with a longer energy-tail validation of shortlisted kernels or
an audited integrator/numerical repair before another serious retained run. Do
not rerun HNN training or sampling unless the target, chart, or HNN question
changes.

The diagnostic complex-to-real casting warnings also remain to be localized
before publication-grade distributional-correlation claims. All current
validity-bearing targets, forces, energies, samples, R-hat/ESS, truth tails,
and direct agreement outputs were finite real float64 where available.

Terminal verification passed with 39 focused tests, syntax and whitespace
checks, 135 JSON parses, and nine top-level hash-ledger replays. The five
serious launches used 9.026 GPU-hours within the 24-hour ceiling. Claude was
healthy on a tiny probe but returned no substantive terminal review output;
advisory external review is therefore unavailable, not passed.

## Superseded Wording

The earlier corrected neural-force terminal result and reset memo said only
LGSSM-KF had a complete matched performance screen. That is historical as of
this repair. PP-UKF, PP-SGQF, and SIR-SGQF now also have complete matched
same-chart exact-gradient comparisons. STR-UKF remains partial for the reasons
above.
