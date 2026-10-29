# Test scalar Philox loop-state layout in the complete value program

The paired streaming full-filter CPU cohort still fails its cost-acceptance
screen despite exact numerical records. The RNG-only component is faster,
and optimized HLO adds three copies/six fusions within the observation body.
Those observations do not establish a cause. Test one bounded intervention:
represent the three Philox uint64 words as three scalar loop-carried tensors,
reconstructing the vector only at the unchanged RNG primitive call and output
boundary. No draw is prefetched or moved across the prediction-validity gate.
The kernel algorithm, Box–Muller transform, carry arithmetic, callback chain,
validity/reset controls, result fields and public signature remain unchanged.

Baseline is the current vector-state streaming authority at ab31c1410; the
candidate edits only state packing in the shared current value program. No
copied algorithm or alternative RNG authority is introduced. The supplied-array
configuration retains its current vector placeholder. This is an execution
layout hypothesis, not a canonical LEDH rebuild or a new production default.

First run the existing eight CPU streaming qualification cases: composed,
annealed, dual-trust, invalid initial, invalid prediction, first-invalid
prediction, time-dependent callbacks and large seeds. They compare the original
post-August21 authorities, exact final RNG state/consumed draws, supplied-array
route, changed observations/seeds, replay and enclosing HLO. Then compare full
complete records against the immediately prior streaming owner in four fresh
CPU processes: T32 vector/scalar and T128 scalar/vector, N64,d2,float64, fixture
seed13, process123, resample17; identical controls from the qualified paired
cohort. Require exact shared outputs including RNG/status. Measure owner+first
call, three conditioning/30 synchronized calls and status/smaps/rusage snapshots
before HLO export/comparison-owner compilation. Use no performance ranking from
this one-process screen. Inspect optimized HLO to determine whether the layout
actually changes loop-copy structure. If it does not, or either horizon is
descriptively slower, stop promotion of this intervention and preserve it as
a diagnostic, rather than launching a large performance matrix automatically.

Only a numerically valid intervention with an explanatory HLO change and
descriptively lower warm time at both horizons proceeds to the existing GPU
qualification and a separately planned matched replicated cost cohort. Neither
gate relaxes the original1.10 upper-bound performance screen. A numerical/RNG
failure stops the intervention for localization. GPU tests use a currently
available non-display device with verified growth, but shared hardware cannot
satisfy uncontended timing/capacity. No training, HMC, environment mutation,
subagents, source-faithfulness/scientific claim or main merge.

Reserve at most8 workers/1800 CPU and600 GPU process-seconds from the remaining
global caps, one numerical worker at a time,300-second initial timeouts. Five
initial CPU workers, a combined readback/policy worker and at most two retries
or a conditional GPU check fit this allowance. Register unique stable-runner
groups and preserve every result under the campaign artifact root. Freeze
runtime sources during the numerical cohort. Any production retention requires
the later replicated cost gate; the current phase is causal localization only.

Skeptical review: the compiler may already scalarize the vector or may recreate
the same copies; a passing correctness test alone says nothing about the
performance mechanism. The CPU baseline must be streaming, not the previously
buffered implementation, and exact complete records prevent a shortened RNG
lifecycle from appearing faster. Reuse the original fixtures and controls;
they are post-invalidation execution fixtures, not scientific LEDH evidence.
The diagnostic is useful even if it rejects this layout hypothesis. Self-review
passes for the bounded intervention and stops above.
