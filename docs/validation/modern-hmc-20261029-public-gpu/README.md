# Public MacroFinance GPU integration — 29 October 2026

**Terminal update:** qualification completed successfully on 29 October at
11:51:16 China time. terminal-review.json records the closeout checks and
settled budget; terminal-inventory.json lists five additional compressed
terminal records. The original launch inventory and cost snapshot are preserved.
See the [terminal result](../../plans/bayesfilter-modern-hmc-public-qualification-result-2026-10-29.md).
Descriptions of launch-only scope below refer to the original packet.

This directory preserves compact evidence for the installed public GPU repair
and fresh qualification launch. It contains no terminal qualification or
posterior result. See the
[engineering result](../../plans/bayesfilter-modern-hmc-public-gpu-result-2026-10-29.md),
[qualification plan](../../plans/bayesfilter-modern-hmc-public-qualification-2026-10-29.md)
and [master](../../plans/bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).

inventory.json records 41 preserved files with original paths, SHA-256 hashes,
original sizes, stored paths and stored sizes. Original paths are relative to
docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5. Most payloads use
gzip; hashes refer to decompressed bytes. JUnit files are stored directly.
The payload includes the exact consumer patch, installation/source hashes,
tests, GPU evidence, public-runner checks, pricing, preflight, launch receipt
and service-launch failure. Payload storage is about 287 kB, excluding indexes.

measured-costs.json records engineering attempts, reserves and remaining upper
bounds. Its qualification status is a launch-time snapshot. The 3600-second
engineering withholding already covers measured engineering. Qualification is
settled once at inclusive supervisor wall time, not again for each child.
Separate SIR funds remain unchanged.

Live campaign:
docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/public-qualification-20261029-01.
Use qualification-01/manifest.json, its native checkpoint and terminal
settlement.json when present for current status. The archived launch.json is
an immutable prelaunch record: its prepared_not_launched field does not
supersede the later receipt or live manifest. Add terminal evidence with new
provenance; do not overwrite the preserved launch records.

Checked scope: source/device/evidence checks, one/four-chain health, mixed-row
isolation, CPU/GPU parity and public XLA execution. Tuning R-hat is reporting-only,
and NUTS is excluded. Posterior validity, ranking, defaults and release readiness
remain unevaluated by this packet.

initial-live-check.json separately records early live progress and verification of all 629 frozen source hashes and 41 preserved payloads. It is a dated snapshot, not a terminal result.
