# AR2 existing-master validation continuation

Structural-09 passed 84 tests (one framework skip), GPU primitive comparisons,
full-consumer precision parity, and compiler call-chain inspection: no targeted
Cholesky/triangular solve custom call remains. Four-position force steady cost:
GPU0 .094–.122s, GPU1 .085–.088s. Largest normalized score change against
structural-07 is 7.11e-5, below the predeclared .002 precision tolerance. These
are single-run engineering results, not a posterior result or speed ranking.

Global charges through structural-09: 30142.368902 CPU-worker and 27448.907342
GPU-wall seconds. Remaining: 23857.631098 CPU / 199351.092658 GPU seconds.
Allocate a new bounded validation pass of 3000 CPU / 5000 GPU seconds from
this unchanged global balance. No transfer of unused old reservations and no
increase of the campaign caps. Prefix:
surrogate-hmc-closure-20260922-ar2-validation-NN. Charge every attempt across
both master prefixes, and charge each subpass only its own work.

1. Re-execute the unchanged four-arm control curve and FP32/FP64 checks on
   calibration/validation data, producing a selection bound to new source.
   Same scope, controls, samples, tolerances and comparator as the controls plan.
   These are engineering requalification data, not a fresh failed-claim retry:
   no posterior claim was run, and claim cost probes do not select controls.
   Allow 180 CPU for checks and 900 CPU / 1500 GPU for the worker.
2. Re-execute complete-transition A/B cost and precision planning under the
   newly issued selection (180 CPU checks, 600 CPU / 900 GPU worker caps).
   Compare the conservative forecast to the actual remaining campaign budget.
   A force-only speedup is insufficient to allocate posterior chains.
3. If affordable, qualify the exact planned uncertainty diagnostic on the AR2
   reference law, with a separately recorded final protocol before launch.
   Then allocate prospective posterior testing from the remaining global budget.

Run the same master entry with --closure-stages ar2-controls, then ar2-cost,
using fresh directories. A completed stage is not a stop. Continue the next
eligible stage. Numerical/control failure is a localized repair trigger;
invalid target/artifacts, absent required checks, or budget exhaustion veto
the affected launch. Independent eligible master work may still continue.

Skeptical audit: small-matrix arithmetic has not qualified the posterior, the
old source-bound selection cannot be reused, and the same four-position
timing is not a matched-precision performance comparison. Baselines, margins,
data and full canonical algorithm remain as declared in the linked controls
and inference plans. This passes for requalification and cost planning.
