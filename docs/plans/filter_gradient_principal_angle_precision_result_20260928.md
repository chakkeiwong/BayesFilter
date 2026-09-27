# Principal-angle SVD repair result

CPU qualification through04582 passes. GPU checks are pending availability;
GPU0 also hosts gnome-remote-desktop despite its disabled display flag, GPU1
has an active display, and both non-display devices were occupied. The scheduler
refused the attempted GPU0 launch before creating a numerical worker. No desktop
fallback is allowed while those busy devices retain ample memory.

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

GPU accuracy/regressions, final policy renewal and evidence sealing remain due.
No full initializer, source-admission or terminal F01--F20 closure follows from
this component result. The unchanged anchor ties and original-controller
trajectory differences remain separate repair work.

04582 versus the saved enclosing fit has 11 strict-bound differences, all in principal-angle diagnostics.

Policy renewal04583 passed160 checks. CPU evidence archive principal-angle-cpu-04583-evidence.tar.gz contains21 verified members, SHA-256`8c5b19842cac8d58507fe30bb1a19b04cf5caa8d47befab916a49da2b2783298`. GPU checks remain deferred due to the recorded occupancy; no GPU numerical run was launched.
