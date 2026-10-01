# Scalar Philox loop state does not pass the performance screen

The scalar-state intervention is rejected and removed from the current runtime.
Both tested horizons have higher descriptive warm times, and the optimized
observation body has three additional copies. Complete outputs remain exact,
including validity and RNG diagnostics. This rejects the particular layout
candidate as a repair; it does not reject streaming or explain the original
buffered/streaming difference causally.

| Horizon, N64,d2,FP64 | Vector-state warm | Scalar-state warm | Scalar/vector | Cold ratio |
|---|---:|---:|---:|---:|
| T32 | 18.447596 ms | 21.331919 ms | 1.156352 | 0.981636 |
| T128 | 38.938178 ms | 86.298757 ms | 2.216302 | 1.014546 |

Each arm used one fresh CPU-reference process, identical fixture13/process123/
resample17 inputs, identical controls, three conditioning/30 synchronized calls,
and the same source/environment/affinity. Runs04828/04829 cover T32 vector/scalar;
04830/04831 cover T128 scalar/vector. Values, every diagnostic and final RNG state
agree exactly. Primary timing/memory exclude HLO export and comparison-owner
compilation. The single-process ratios are nomination screens, not statistically
supported speed rankings. CPU remains an explicit reference.

CPU04827 passed all nine existing streaming/callback checks, including healthy,
invalid, time-dependent and large-seed cases. Final draw counts/states match
the independent original authority, and no full process-noise buffer appears.
The complete-filter readback04832 passes161 checks. Optimized observation-body
unique copy counts are360 vector versus363 scalar; fusion and while counts
are unchanged at602 and25. These are static counts, not executed multiplicities
or causal runtime proof. The candidate failed the predeclared nomination screen,
so no GPU trial or larger matched matrix was launched.

The runtime was restored byte-for-byte to the qualified vector-state baseline:
SHA256 `150c30a6970dc6e69b053c7ee24308feab4185112c2ed07e95a773810f51df68`.
The rejected candidate hash is
`c7e428521ed7569c423f400008708d2cf8b575f79c9229602d963a564a5c0a39`.
Its source, launch harness, exact outputs, HLO and manifests remain only in
the64-member verified archive `streaming-state-layout-04832-evidence.tar.gz`,
2,994,076bytes, SHA256
`f3df7d4aebbb431200e9d18a2a5052ae014376605689cf4ba246aa735326356a`.
Trial-only launch groups/harnesses were retired; existing streaming regression
coverage remains. Retirement04833 passes161 archive/restoration/policy checks.
No algorithmic, RNG, tolerance, default or allowlist change remains from this
intervention. All seven workers passed their correctness/artifact checks and
used202.332048 CPU process-seconds, zero GPU, within8 workers/1800 CPU/600 GPU.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Reject scalar-state layout | Complete parity passes; cost screen fails | Both warm ratios exceed1 | One process per arm; scheduling effects | Restore and archive, no expansion | Statistical ranking or copies alone caused slowdown |
| Keep original streaming cost gap open | Prior paired upper bounds still exceed1.10 | Original cost acceptance blocked | Full-filter/compiler interaction | Review another specific mechanism only with evidence | Restored vector runtime now meets cost gates |
| Continue confirmed consumer repairs | Fitted APF/iAPF host recurrences found through real endpoints | Those routes remain unqualified | Live RNG compatibility and adaptive shapes | Execute bounded preparation in fitted-APF plan | Whole-repository completion |

Self-review: the result is not an XLA arithmetic failure or RNG mismatch. The
strongest alternative explanation for the observed cost gap is process-level
timing noise or compiler scheduling beyond copy counts. The predeclared rule
permits stopping this candidate without proving which explanation dominates.
The negative evidence is preserved; thresholds were not relaxed.
