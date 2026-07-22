# PP-UKF Energy Veto Correction

Date: 2026-07-22

Verdict: the previous energy veto was wrong.

Claimed target: native HMC divergence or another declared numerical invalidity
that can stop candidate warmup.

Quantity actually computed: the number of finite `log_accept_ratio` values
below `-1000`, plus any non-finite values. A finite very negative log acceptance
ratio means the proposal has extremely low Metropolis acceptance probability.
It is not equal to a native divergence flag and does not prove an invalid HMC
transition.

Correction:

- non-finite state, target value, or log acceptance remains a hard veto;
- invalid PP-UKF target status remains a hard veto;
- no chain movement remains a hard veto;
- positive native divergence is a veto only when the kernel exposes it;
- finite `log_accept_ratio < -1000` is renamed
  `extreme_log_accept_count` and is explanatory only.

The historical `energy_error_divergence_count` field remains temporarily as a
compatibility alias, but its definition now states that it counts finite extreme
log acceptance values and must not be used as a health veto.

Consequences for `attempt-04`: L=5, L=9, and L=12 are not rejected. Each has
only a 1,000-transition warmup prefix, below the required 2,000 minimum. All
three therefore remain unevaluated and require continuation or a clean rerun
under the corrected controller.
