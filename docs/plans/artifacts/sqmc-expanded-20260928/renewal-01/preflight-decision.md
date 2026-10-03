# Preflight decision

Proceed with the authorized expanded comparison. The 47 focused CPU regressions
passed, and the bounded GPU preflight passed 16 graph/XLA parity cases across
P44 d3 and full A/Q d3/d10. The maximum absolute score difference was
2.842170943040401e-14. This establishes backend parity on these small mechanics
fixtures, not Kalman score accuracy or method superiority.

Full d10, T2, N1020 timing used all 157 coordinates, flows 2/8, and IID and
permutation routes. Warm repeat times were 5.10–8.56 seconds; cold times were
15.00–26.18 seconds. The complete-ladder estimate is 25,190.83 seconds, or
37,786.24 seconds with the predeclared 50% margin. Remaining aggregate GPU
allowance is 40,559.57 seconds. The fixed elapsed deadline is
2026-09-28 07:25:28.470267 UTC. These bounds permit launch but do not guarantee
completion; actual wall time and stop conditions remain enforced.

RTX 4080 SUPER was idle before the check. GPU memory growth was verified,
TF32 disabled, FP64 XLA used, and allocator peak was 69,390,592 bytes.
The preflight consumed 722.43 seconds; prior expanded work is charged 1,918
seconds conservatively. Full evidence is in gpu-check-01/result.json and
manifest.json, complete output in gpu-check-01.log, accounting in budget.json.

Skeptical audit: the checks answer mechanics and cost only. The final runner
retains scope-specific tuning, the matched full-horizon Kalman oracle, all
coordinates, the complete Cartesian final seed design, conditional heuristics,
and candidate-invalidity versus infrastructure-failure classifications. No
unresolved preflight failure remains. Preserve all completed units if a later
localized repair is needed. The terminal scientific comparison remains pending.
