# M13 weighted-moment candidate result — 2026-09-24

The weighted-first arithmetic removes the nonfinite fourth-moment targets in
the two failed rows. GPU/XLA primitive comparisons pass, including analytical
weight and particle derivatives and zero-weight extreme particles. Independent
FP64 discrepancies are below9.3e-8 under the declared normalization. The master
review passes42 tests, including existing full-correction JVP checks.

The candidate does not close the filtering failure: both rows still fail the
higher-moment correction at time0. Targets are now finite (last-coordinate
fourth moments3.1273e19 and1.4723e19), but the aggregate correction validity is
false. All seven previously healthy rows remain valid. Their largest absolute
log-likelihood change is0.0448747; the normalized value screen fails at0.02748,
and normalized score change reaches0.00682. This is a diagnostic trigger for
inspection, not a material posterior error finding. No parameter or posterior
comparison has been run for this change.

Evidence: `artifacts/surrogate-hmc-closure-20260922-m13-controls-11/`. Exact
command, source snapshots, unchanged input hashes, memory policy, primitive
and complete consumer diagnostics are retained there. Only the declared shared
moment helper source changed in the canonical closure. The implementation is
an evaluated arithmetic candidate; it has not been promoted as a complete fix.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Incomplete repair | Primitive parity passes; full-row repair fails | Typed correction failure; healthy-case screen fails | Which correction substep remains invalid | Preserve computed substep validity flags and replay | No HMC or posterior readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Two typed finite-program failures persist |
| Statistically supported ranking | None |
| Descriptive-only differences | Healthy-row likelihood/score shifts; timing |
| Default-readiness | FP32 default retained; candidate repair not promoted |
| Next evidence needed | Exact correction substep validity and then a discriminating repair |

Charge86.201810 CPU/41.070011 GPU seconds. Cumulative45263.767463292 CPU/
35245.301898455 GPU seconds. Controls pool remaining111.242237 CPU seconds;
global8736.232537 CPU/191554.698102 GPU seconds. Post-run alternative: a later
trust-cap derivative or cloud restoration can fail even when weighted moments
are finite. The aggregate flag cannot distinguish these explanations. Emitting
already-computed validity flags changes no accepted numerical result.
