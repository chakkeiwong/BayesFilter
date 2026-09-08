# Phase 28 Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| 1 (`phase28-fresh-theta-pilot`) | none | none | `PASS_THETA_MEASURE_PILOT` for paired C0/M0 |

The receipt is
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot/`.
It completed in `101.57 s`; both arms had finite/status-valid 64/64 theta
rows, reached beta one, and passed all density/protocol gates. Terminal ESS
fractions were `0.9494` (C0) and `0.9796` (M0). Weighted negative-mode
fractions were `0.5007` and `0.7125`, respectively, and are descriptive only;
the paired arms use different proposal laws and no uncertainty ranking is
valid. The geometry file was hash-bound as a calibration warm start and no
old particles were loaded. Phase 29 is refreshed to consume only the M0
receipt's `[64,4]` theta bank.
