# SVX analytic-route diagnostic result

Date: 2026-08-06
Status: `ROOT_CAUSE_DIAGNOSED`

## Question answered

We tested the two competing hypotheses for the failing SVX-ZC serious route:

1. could an existing analytic transformed-SV route be wired in as a drop-in replacement?
2. or is the current active SVX target an adjacent-state frozen-core finite program
   that still lacks an admissible analytic derivative backend?

The tests show that the second explanation is the correct one.

## What was tested

### Active route probe
The current active route from:

- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`

was evaluated on a small probe batch and under prefix truncations.

Observed facts:

- `T=1` is finite and unchanged when the one-axis initial core is zeroed.
- `T=2` changes when the adjacent frozen UKF cores are zeroed.
- Zeroing the adjacent cores causes the `T=2` value path to collapse to `-inf`.

This means the active value program is **genuinely adjacent-state frozen-core dependent**.
The one-axis initial core is identity-bound but not the numerical bottleneck.

### Candidate analytic route probe
The strongest existing analytic transformed-SV candidate from:

- `bayesfilter/highdim/sv_mixture_cut4.py`
- `bayesfilter/highdim/filtering.py`

was compared on the same probe point and prefixes.

Observed value mismatch (active minus candidate):

- `T=1`: about `+1.67e-2`
- `T=2`: about `-9.15e-2`
- `T=3`: about `-1.02e-1`
- `T=10`: about `-4.998e-1`

The mismatch grows after the first adjacent-state step, which is exactly where the active route switches into the UKF-frozen adjacent-core branch. So the candidate analytic route is **not** the same finite program.

## Root cause

The failing serious SVX-ZC lane is blocked because:

1. the active target still uses an autodiff-backed score through the fixed TT program; and
2. the repo does **not yet have an analytic derivative for the exact adjacent-state frozen-core finite program used by the active target**.

So the missing artifact is **not** a simple wiring fix.
It is the analytic adjacent-state derivative of the current active SVX finite program.

## What is not the fix

- swapping in the transformed-SV analytic fixed-branch TT route as if it were the same target;
- keeping the current autodiff route and hoping memory behavior improves;
- changing the target silently without a versioned contract change.

Those are all wrong relative to the current target.

## Next implementation step

Implement a **new analytic score backend for the current SVX adjacent-state frozen-core finite program**, preserving:

- the current transformed-observation actual-SV semantics,
- the current UKF-frozen adjacent-core initialization identity,
- the current target contract / scope discipline,
- but replacing the HMC-facing score with an admissible analytic derivative backend.

That is the mathematically correct next step.

## Nonclaims

- No HMC convergence claim.
- No production-readiness claim.
- No claim that the existing analytic transformed-SV route is wrong in general; only that it is not the same target as the active SVX lane.
- No claim that the final analytic implementation already exists in the repo; the evidence suggests it does not.
