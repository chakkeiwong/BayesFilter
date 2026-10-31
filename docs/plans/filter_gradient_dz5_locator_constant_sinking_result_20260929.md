# DZ5 constant-sinking intervention result

Disabling XLA's named while-loop constant-sinking pass changes the original
locator to reproduce the ordinary candidate exactly in the complete short
record and every callback byte. This establishes compiler-pass sensitivity in
the frozen historical target. It is not a portable repair: disabling the pass
in both original and candidate still leaves46 differing score fields.

| Arm | Run | Known214/217-instruction fusion counts | Computations |
|---|---|---|---|
| Original normal |04877|26 /0|17580|
| Candidate normal |04878|22 /4|17577|
| Original pass disabled |04879|22 /0|17854|
| Candidate pass disabled |04880|22 /0|17851|

Both normal controls reproduce04656/04657 exactly, including all callback
arrays and short records. Their identical-input callback row1 scores differ
by at most1.52767e-13, followed by different positions and values. Original
pass-disabled matches candidate normal at all six callback rows and every
result field. The two pass-disabled arms have identical positions, values
and validity; only row3 scores differ, by at most4.82743e-11. This later
difference propagates to46 short-record fields. All arms retain three actual
optimizer objective batches, one trace and no host callbacks.

The intervention removes all four217-instruction signatures, but it also
changes hundreds of other computations. In each disabled arm the four
corresponding original214 signatures are absent too. Opaque printed constant
sites change3033→2965. Thus the evidence does not isolate one multiply, prove
literal equality at opaque sites, or identify the executed machine code.
Independently exported IR may compile separately. The initial four-fusion
lead remains narrower than a generic dtype explanation, but a module-wide
pass switch does not resolve the whole discrepancy.

Readbacks04881/04882 each pass179 checks. The final report additionally compares
scores at every identical-input row, preserving the earlier report unchanged.
It verifies generated child bytes, environment, frozen snapshot, complete
loaded-source identities, dispatch, callback/HLO hashes and source controls.
The existing parser and policy checks pass; new diagnostic code passes Ruff
and whitespace checks. Runtime source, default compiler settings, numerical
tolerances, RNG streams and policy allowances are unchanged.

The six-worker unit uses1571.533909 CPU process-seconds and zero GPU, below
its4200-second cap. Remaining global budget is24.562895 CPU /24.770371 GPU
process-hours. No worker remains active. Export wall time and host memory are
instrumentation, not production costs. All six runs and their exact commands,
inputs, full records and IR remain under the existing raw artifact root.

Skeptical terminal review: the supported conclusion is a causal effect of the
named pass on the observed frozen short program, with broader context still
confounding a local arithmetic explanation. It would be wrong to install the
flag as a default or claim that identical first-step scores repair the full
optimizer. The121 full-trajectory differences and both unconverged optimizers
remain open. A further locator experiment must isolate local gradient
arithmetic or emitted lowering while preserving controls; no unrelated flag
search or unchanged full trajectory is justified.

Continue the independent Gaussian analytical model-binding repair in
`filter_gradient_gaussian_model_binding_20260929.md`. This avoids spending more
of the campaign on a historical compiler diagnostic while confirmed active
execution gaps remain. Adaptive iAPF is deferred. No independent reviewer,
canonical LEDH claim, current-source locator admission or main merge follows.
