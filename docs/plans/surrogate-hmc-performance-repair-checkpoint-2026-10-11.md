# Surrogate-HMC computational campaign checkpoint

Updated 2026-10-21. The active reset memo is
[surrogate-hmc-five-model-n4032-reset-2026-10-11.md](surrogate-hmc-five-model-n4032-reset-2026-10-11.md).
The [final master summary](surrogate-hmc-master-final-summary-2026-10-21.md)
records the terminal sampling outcome and the completed computational repair.

Four models passed their sampling rule. SIR ended its bounded run during
warm-up with no retained draws. The native batching implementation and capacity
campaign are complete; SIR NeuTra adapter/update work, training and downstream
sampling remain pending. The percentage-only numerical blocker is retired by
[sir-batching-materiality-policy-2026-10-21.md](sir-batching-materiality-policy-2026-10-21.md).
No new GPU run is part of this closeout.

The current task is the user-authorized commit, main/remote integration and
branch synchronization. Detailed earlier execution and budget records are
preserved in artifacts/surrogate-hmc-closeout-20261021-01/before-surrogate-hmc-performance-repair-checkpoint-2026-10-11.md
and the linked terminal artifacts. The next research implementation task remains
the existing parent plan's Phase 1 adapter and whole-update checks; it is not a
mandatory precision-localization campaign.
