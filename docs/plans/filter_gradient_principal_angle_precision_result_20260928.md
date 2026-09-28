# Principal-angle SVD repair result

CPU qualification through04583 passes. GPU accuracy checks 04599/04600 and
all 48 regressions in 04601 now pass on scheduler-selected eligible GPU3.
Earlier occupancy correctly deferred GPU work: GPU0 hosts remote desktop despite
its disabled display flag and GPU1 has an active display. No desktop fallback
was used.

The runtime uses explicit binary64 XLA SVD convergence for the padded overlap
matrix. The mathematical principal angle, rank selection, unit-value snap and
acceptance thresholds are unchanged. The branch remains unmerged.

| CPU kernel | Singular-value accuracy | Cold seconds | Warm median ms/call | Warm RSS MiB | HLO bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Before04579 | Fails; max error7.73e-8 | 5.9181 | 9.4619 | 876.71 | 879620 |
| Repaired04580 | Passes; max error8.99e-15 | 5.8004 | 11.3188 | 872.55 | 880771 |

These are descriptive measurements from separate fresh workers, with five blocks
of20 timed calls each. The defective baseline cannot support a performance
ranking. Repaired warm time is descriptively19.6% higher. Graph size increases
by13 bytes and HLO by1151 bytes; post-cold host RSS increases only1.35MiB over
100 warm calls. The cold compiler allocation is roughly302MiB. This probe finds
no new continuing warm-memory growth, but does not prove native cache eviction
or the complete initializer's memory lifetime.

Analytic rotations/rank changes, existing complete stability records and
factor/dense derivative regressions pass48 CPU checks in04581. The strict
1e-10 full-record angle comparison still differs by up to1.61e-9 degrees on
the actual nearly aligned subspaces, even though their cosines now agree to
binary64 precision. That mismatch remains reported and has not been waived.

Controlled CDF fit replay04582 passed on the identical04576 operand hashes.
The following saved comparison records the remaining changed fields; acceptance,
selected geometry and factor fits are not altered by this SVD-only repair.

No full initializer, source-admission or terminal F01--F20 closure follows from
this component result. The anchor tie repair and original-controller trajectory
differences remain separate work under their linked September 28 plans.

04582 versus the saved enclosing fit has 11 strict-bound differences, all in principal-angle diagnostics.

Policy renewal04583 passed160 checks. CPU evidence archive principal-angle-cpu-04583-evidence.tar.gz contains21 verified members, SHA-256`8c5b19842cac8d58507fe30bb1a19b04cf5caa8d47befab916a49da2b2783298`. The later GPU evidence is recorded below.

GPU continuation 04599/04600 uses physical device
`GPU-b8045e28-4433-ec7a-77a5-db0636748322`, float64, TF32 off, XLA and verified
memory growth. The scheduler permits correctness checks on this shared idle
device; its `performance_preflight_uncontended` field is false. These are
descriptive observations only.

| GPU kernel | Maximum singular error | Cold seconds | Warm median ms/call | Warm RSS MiB | Allocator peak bytes |
|---|---:|---:|---:|---:|---:|
| Before, 04599 | 7.7343e-8 | 9.4270 | 75.7345 | 1105.02 | 70,400 |
| Repaired, 04600 | 1.5543e-15 | 9.3036 | 92.2981 | 1100.38 | 70,400 |

The repaired GPU solver passes the declared singular-value accuracy gate.
The strict angle-record comparison still fails by up to 1.9893e-10 degrees,
which remains recorded separately. HLO sizes are 879,620/880,771 bytes and
allocator current bytes are 10,240 in both arms. No new device-memory increase
appears in this kernel probe. Warm duration is 21.87% higher, crossing the
20% investigation trigger. Tighter SVD convergence is the concrete code change,
but these shared-device runs do not establish how much time it caused. Keep
matched uncontended cost attribution open; do not infer performance superiority
or weaken numerical accuracy to satisfy timing. Full consumer lifetime and
terminal cost qualification remain separate.

GPU runs04599--04601 are preserved in
`artifacts/filter-gradient-repair-20260917/callback-boundary-gpu-repairs-04605-evidence.tar.gz`,
64 verified members, SHA-256
`88f352b5d727897d681ae84610ab18c434ed49da2c5063b4731652689f3231ed`.
Final policy04605 passes160 checks before the separate exact-input fit harness
was added. The SVD allocation closes at16 workers without new global budget.
