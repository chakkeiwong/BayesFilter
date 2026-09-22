# Replay metadata check repair

The first d5 numerical replay completed, but its checker rejected
`identical(a$data,b$data)`. Inspection showed the observations, transition
matrix, learner histories, final results, guides and fit diagnostics were equal;
the new CLI parses the seed as an R integer while the prior literal was a double.
Thus the rejected condition compared metadata storage types as well as data.

Repair: require exact observations and transition matrices and equal scalar
seed values. Keep `identical` for all algorithm outputs, stopping histories,
guides and diagnostics. Do not loosen any numerical tolerance. Rerun in fresh
versioned directories and preserve both failed and successful attempts. The
4.965 worker seconds consumed by the initial replay/checker remain charged.
This is a localized checker failure, not a candidate rejection or a change in
the scientific contract. Also verify the new summary helper's central statistics
against Python standard-library calculations on the preserved phase15 records.
