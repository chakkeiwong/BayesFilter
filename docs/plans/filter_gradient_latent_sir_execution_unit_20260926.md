# Latent SIR simulation execution repair

Question: can the public fixed-noise latent SIR simulator default to one XLA
program while preserving its existing latent, physical and observation paths,
initial clipping convention, parameter scaling and rejected-input behavior?
The transitive audit found both a Python time loop and non-graph-compatible
scaled-model construction/time conversions. A loop-only edit is insufficient.

The target is the existing `LatentPreclipSIRSSM` representation, explicitly
classified `extension_or_invention`, with no canonical Contract E admission.
This repair does not change its scientific target or claim source-route
faithfulness. Before implementation, the following anchors were inspected:
Zhao--Cui paper section 6.3, equation (37), extracted text lines 2250--2312;
author sources under `third_party/audit/tensor-ssm-paper-demo/models/sir_austria`.
Extract `matlab/document.xml` from the `.mlx` files and number lines of
`''.join(tree.itertext())`: `priorsam.mlx:2` samples an un-clipped Gaussian
initial state; `st_process.mlx:2-3` adds process noise, then clips susceptible
coordinates; `ob_process.mlx:2` adds observation noise after state selection;
`sir_step.mlx:5-18` uses four 0.005 steps and a half-step fourth RK stage.
Preserve these existing local choices. The local parameter chart and pre-clip
latent density remain adaptations, not author filtering-algorithm claims.

Use the shared `SpatialSIRSSM` numerical authority for rate-scaled transitions;
do not copy its RK4/RHS into a simulator fork or replace a value program with
an analytical-score helper that performs extra derivative work. Separate
validated static model construction from tensor parameter operands. Preserve
the same finite RK stage expressions and covariance symmetrization. Audit
scaled-model validation before changing it: overflow, nonfinite and underflowed
nonpositive rates must retain explicit rejection, not become silent outputs.
If this needs a shared parameterized-rate core, keep existing value and tangent
consumers wired to their current authority and verify those consumers.

Expose a stable owner with JIT on by default; graph is an explicit reference
option. Cache/retain owners only for declared fixed configuration and pass
theta and all supplied noise tensors as operands. Test changed operands and
one trace. Use native time control and preallocated output storage, including
the T=0 case without tracing an invalid empty-noise slice. t=0 must remain
un-clipped even when susceptible values are negative; t>=1 clips susceptible
coordinates only. Noise order, matrices, parameter order and all three returned
path shapes/keys remain unchanged. No new random stream, ridge, clipping rule
or model default is authorized.

Primary gates use fresh frozen Git source from the commit recording 04037,
not historical LEDH results: J=1,2,9, T=0,1,3, both RK variants where supported,
identical supplied noise and changed theta/noise. Use the existing small-model
and source-style path fixtures as independently executed authorities. Include
negative initial susceptible states, process noise crossing the clip boundary,
negative infectious states (never clipped), malformed shapes and invalid
parameter controls. Keep FP64 bounds at 5e-12 absolute/relative for paths;
discrete decisions, shapes and exact replay remain exact. Preserve and localize
any failure rather than relaxing a comparison. Existing analytical derivative
and general simulator consumers must pass after a shared-core edit.

Measure original eager, explicit repaired graph and repaired XLA in fresh
CPU/GPU processes with 20 exact replays. Attempt and record the original
enclosing-XLA boundary separately; if it cannot compile, do not invent its
timing or silently patch the frozen baseline. Record cold/warm time, host RSS,
TF allocator current/peak, graph/HLO size, trace count, source bytes, hardware
and memory-growth provenance. Share-aware GPU results are descriptive only.
These fixture costs cannot close target-scale capacity or repeated-constructor
native memory retention.

Budget inside existing caps: at most 16 CPU / 12 GPU supervised invocations,
3600 CPU / 3600 GPU process-seconds. Through 04037 the remaining budget is
32.488836 CPU / 30.624976 GPU hours, already including the user's extra CPU
allocation. Use the stable campaign runner and versioned artifacts, one worker
at a time, with numerical sources frozen and snapshotted per attempt. Stop an
affected comparison on source drift, invalid evidence or exhausted budget;
localize numerical/compiler failures within the unchanged contract.

Skeptical review: the principal hazards are clipping x0, changing process-noise
matrix orientation, dropping scaled-model guards, recomputing RK substep ratios
inside XLA, forking the RHS, and treating an uncompiled original as an XLA
baseline. The anchored clipping checks, frozen records, explicit static/tensor
boundary and existing-consumer checks address these hazards. This is an
execution-only repair plan. It does not admit canonical LEDH, Zhao--Cui
production filtering, HMC, posterior correctness or the whole master program.
Implementation may proceed after confirming the exact validation contracts;
no new approval is required inside the existing campaign.
