# Fresh M13 controls at nominated N4096 — 2026-09-24

This continues the surrogate-HMC closure master after the bounded particle
ladder. The question is whether N4096 supports a finite stable region of the
existing fixed-reset/cap LM control curve on independent datasets. It does not
change the model, prior, full canonical algorithm, FP32/TF32/XLA default, or
final posterior comparison. LG1 remains closed and Claude deferred.

## Evidence contract

Run all four LM multipliers1,.1,.01,.001, preserving reset ridge1e-5, trust
radius.1, pairwise cap2 and coordinate cap16. Each candidate uses N4096 and the
repository-issued K2048. Controls are M13 warm starts, not accepted defaults.
Use fresh calibration seed1704/design9 and validation2704/design10 (attempt4).
Preserve claim3701/design2 and never evaluate it in tuning. The same nine
nominal and nine stress theta points are used; these are prior-based diagnostic
situations, not asserted posterior coverage. Full consumer diagnostics are
required, including failure flags and substeps.

Keep the existing engineering selection rule: first adjacent pair with healthy
nominal rows, flagged stress failures, no new stress failures under stronger
protection, and normalized change<=.001 on both partitions. The selected object
would be controls for an imperfect proposal-field experiment only. It cannot
issue exact-score, posterior, endpoint, HMC or default readiness. Expected
candidate failure triggers further assessment, not rejection of surrogate HMC.
Wrong input/source/device, unflagged nonfinite output, missing required evidence
or budget expiry veto continuation of this attempt. Save completed arms if the
attempt stops. No change in saved-run warning policy is introduced.

Important limitation: controls-12 changed scores by .011 in the normalized
saved-run metric after diagnostic outputs changed XLA compilation, with identical
values/moments. Thus .001 is an engineering nomination screen that may be below
FP32 compilation variation. A failed curve alone cannot establish material harm.
The curve is measured to assess this scope; no threshold is selected on its
result. If no pair passes, diagnose validity versus numerical variation and
posterior relevance before proposing a revised selection rule. Do not promote
or reject the scientific method using that proxy.

## Audit, budget and command

N4096 is nominated by one design, so fresh partitions are essential. All
existing candidate settings remain explicit hypotheses. Primary screen and
explanatory diagnostics remain distinct. The exact model posterior reference
is not yet qualified; it is not a comparator in this preparatory stage. A
heuristic-dominance or efficiency ranking of surrogate HMC remains NOT_EVALUATED.
Pre-mortem: a finite control curve could hide biased finite likelihoods; it
cannot authorize posterior claims. A failed curve may reflect FP32 arithmetic
rather than a material control effect. These limits and full diagnostics make
the bounded engineering experiment interpretable. Skeptical audit passes under
these nonpromotion limits, not as evidence the inherited threshold is correct.

Allocate2000CPU/2400GPU from the existing balance8324.426CPU/191401.729GPU,
as m13-controls-recovery. No total ceiling change (54000CPU/226800GPU).
One review100CPU/120wall and one GPU worker1600CPU/1800wall; up to8 control arms,
each evaluating nominal and stress. A localized harness retry may use the
remaining subpass balance and a fresh versioned root.

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-controls-recovery --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-recovery-01
```

The master archives sources, command logs, environment/device/memory settings,
seeds, data, controls, conditional diagnostics, result and accounting. This is
part of the existing master, not a change of research direction.
