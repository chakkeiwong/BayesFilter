# Comparator Correction

Date: 2026-07-22

The original `result.json` was produced before a source audit of the generic
fixed-SGQF comparator. Its `fixed_sgqf` diagnostic value/score must not be used
for this row. `tf_fixed_sgqf_filter` propagates before every observation,
including `y0`; the canonical predator-prey T20 target assimilates `y0` from the
initial law before the first transition. The number is therefore wrong relative
to the declared target, not merely a low-accuracy approximation.

The GenUT, bootstrap-PF reference, and principal-square-root UKF quantities in
the original artifact are unaffected. The original artifact is preserved for
provenance. The active leaderboard runner now fails closed with a reason that
names this timing mismatch. The historical retained-grid Zhao-Cui result
remains diagnostic/historical and is not an oracle.

Original result SHA-256 is recorded in `run_manifest.json`. This correction does
not admit GenUT: an independent marginal-score truth authority and leaderboard
integration are still missing.
