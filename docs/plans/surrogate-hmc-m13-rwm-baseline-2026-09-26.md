# M13 classical sampler comparator within the approved continuation

The sole target is the same frozen full-support4096x8 importance endpoint plus
the independent N(0,log(2)^2) priors used by the running M13 surrogate HMC.
This independent CPU reference comparator is part of the existing master.
It does not select HMC controls or change the endpoint.

Construct a Gaussian random-walk Metropolis baseline. Four prior-scale starts
match the public HMC pilot. Use512 discarded scale-adaptation transitions with
prior covariance and initial scale0.5; estimate covariance from the last256
draws of each chain and shrink it10% toward the prior. Run512 further discarded
scale-adaptation transitions with this covariance, then freeze it and the scale.
The Robbins-Monro scale gain is(i+10)^(-0.6), targeting0.25 acceptance. These
are baseline tuning hypotheses; fresh-chain diagnostics can reject them. The
positive prior mixture has a derived covariance lower bound0.1log(2)^2 I.
No target ridge, force substitution or acceptance approximation is introduced.

Restart four fresh chains at the declared starts, discard500 fixed-kernel
warmup draws, and archive eight250-draw retained chunks. This explicitly
classified reference route cannot issue HMC tuning authority. Use the same
two-look posterior assessment at1000 and2000, including R-hat, ESS, MCSE and
unchanged materiality margins. Every chunk must have finite values, all-chain
movement and valid endpoint evaluations. Invalid proposals are rejected and
flagged; any such flag prevents qualification of this pilot.

Direct finite-grid IID draws and integration supply the other simple
comparators. Report state-conditional RWM acceptance in nominal, intermediate
and outer prior-scale regions; an unvisited region has no conditional sampler
evidence. Passing establishes only a viable classical comparator. A single
CPU RWM pilot and GPU HMC pilot do not support a hardware-fair speed ranking.
Material posterior disagreement or poor mixing requires repair; it cannot
be hidden by high acceptance or used to claim HMC superiority.

All arithmetic is TensorFlow/TFP; stable signatures compile the MH loop with
XLA. CPU execution is an explicit independent-reference exception with GPUs
hidden before import. Archive every warmup and retained trace, configuration,
seeds(20260926,2810) with distinct fold-ins, source hashes and command.
Charge tests, compilation, failed attempts and sampling to the existing
6000CPU/6000GPU baseline allocation, less all assessment-development receipts.

Skeptical audit: comparison target/data/prior/precision are identical; the
covariance changes proposal geometry only and is frozen before retention;
the fresh warmup is excluded; baseline failure is not direction failure.
Gaussian sampling, deterministic replay and invalid-endpoint rejection tests
must pass before the model run. No numerical or efficiency result exists yet.

Budget clarification: the6000CPU allowance is shared with materiality-control
attempts01/02, whose1274.239627CPU seconds must also be deducted. The launcher
now reads those original receipts as well as every assessment receipt. The
materiality-repair attempt belongs to the separate10000CPU repair allocation.
No stage receives an extra6000CPU allocation. Run development03 next, followed
by reference-coverage02 only if all focused checks pass.
