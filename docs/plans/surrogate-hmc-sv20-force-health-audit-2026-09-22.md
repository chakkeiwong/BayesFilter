# SV20 original-force status supervision

The active SV20 B/R plan requires healthy original force evaluations on every
retained chunk. A call-chain review found that the shared public archive
checker checks finite states/energy, endpoint validity, movement and divergence,
but intentionally does not reject a deterministic position-only force extension.
That is consistent with the generic surrogate-HMC mechanism; it does not
enforce SV20's additional original-force condition.

The sampler already preserves `force_fallback` on every transition. Run a
separate CPU-only status audit against checksum-verified tuning and retained
archives as they become available. Any retained fallback triggers a health
veto and stops the current attempt. Missing/corrupted telemetry prevents
accepting a result. Warmup and verification counts are preserved as additional
diagnostics. This uses an existing criterion, changes no numerical code or
statistical look, and does not require force accuracy or equality with the
endpoint gradient. A qualified deterministic extension may be studied under
another explicit scope; it is not the original-force condition frozen here.

Keep the active run and its bound sources unchanged. After it finishes, add
the explicit consumer-level guard before interpreting retained samples in the
SV20 runner, with a focused regression for a finite fallback. The generic
public tuner must continue to distinguish a qualified extension from an
invalid endpoint or nonreversible map.

Reserve30 CPU seconds and no GPU work within the current SV20 family/global
envelope. This is a read-only numerical-status diagnostic, with GPU devices
intentionally hidden. Record exact command, environment, source hash, status
counts, manifest hashes, elapsed time and incremental CPU accounting in
`surrogate-hmc-all-models-20260922-sv20-force-health-audit-01`. Poll no faster
than every5 seconds; stop after the inference parent writes final accounting
or the audit detects an invalid archive/fallback. Supervision ends no later
than the active inference's existing12600-second wall reservation.

Skeptical audit: this is a hard status check, not a new posterior look or a
post hoc change to MCSE/equivalence rules. Zero fallback counts establish
that this omitted guard did not affect this particular run. They do not fix
the missing guard in future invocations or prove global force validity.
