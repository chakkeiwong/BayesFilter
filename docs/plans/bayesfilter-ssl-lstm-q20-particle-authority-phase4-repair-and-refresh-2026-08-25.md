# Phase 4 Repair and Refresh Note

Status: `REPAIR_TRIGGERED_LONGER_SCOPE_SCREEN`

Classify a failed training run before changing anything: device/policy,
batch-shape, transport/Jacobian implementation, target/status, tuning, or
scientific candidate. Repair the smallest class first. A poor whitening score
does not justify changing the authority or claiming that NeuTra is invalid;
the two-mode transformed-target and exact parity gates decide whether the
training component is usable.

## Attempt classification

Attempts 1-2 failed before model execution (`bayesfilter` import path, then
GPU memory-growth configuration order). These were harness/policy-order defects
and were repaired without changing the target or criteria. Attempts 3-5
reached the runner but failed the audit partition design (one-sided metadata,
then no two proposal components); the split was repaired to a deterministic
signed-coordinate stratification. Attempt 6 then passed all hard engineering
and transformed-target status gates on GPU/XLA.

The remaining finding is explanatory/tuning evidence: both arms clipped their
gradient on all three updates, and the selected compact arm had validation
latent `max |mean| = 1.3370` and maximum off-diagonal covariance `4.0324`.
This is insufficient whitening, but the run was only a three-update screen;
it is not evidence against the transport identity or the research direction.

## Repair action

Run a same-scope longer screen with the controls, seed, split, target, GPU
policy, and two arms frozen, changing only the update budget from 3 to 20.
This discriminates under-training from an immediately unsuitable empirical
measure at low cost. The repair remains candidate evidence; no arm is ranked
or promoted from one seed. If the longer run still clips and fails to reduce
the latent diagnostics while exact gates remain valid, classify the issue as a
tuning/data-quality finding and refresh Phase 5 with a fresh target-specific
training campaign rather than launching HMC.

## Actual repair result

The longer repair is recorded as `phase4-attempt7` below after execution. The
terminal decision must report hard gates separately from the whitening result.

## Actual repair result

`phase4-attempt7` passed the same GPU/XLA and transformed-target hard gates
after 20 updates per arm. The compact arm's validation loss fell from `17.45`
at update 3 to `16.45` at update 20; its latent mean diagnostic fell from
`1.3370` to `1.2440`, and the largest off-diagonal covariance fell from
`4.0324` to `3.7309`. Gradient clipping was still active during most of the
run. These changes are descriptive and remain far from an IID Gaussian
criterion.

The repaired mutation bank was then screened in
`phase4-mutation-revalidation-attempt1`. It passed the exact and status gates,
but its three-update compact arm had latent mean diagnostic `3.2681` and
off-diagonal covariance `2.3844`; this does not establish better whitening.
Both branches therefore remain role-limited candidate screens, and no HMC
campaign is justified by this evidence.

Status after repair: `PASS_HARD_GATES_ROLE_LIMITED_TUNING_UNRESOLVED`.

Refresh Phase 5 with a strict non-HMC boundary and state whether a separate
sequential-HMC plan is justified, blocked by missing authority, or unnecessary
because the training candidate failed an exact gate.
