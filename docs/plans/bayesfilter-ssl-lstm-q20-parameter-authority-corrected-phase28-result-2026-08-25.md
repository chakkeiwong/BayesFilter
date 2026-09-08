# Corrected Parameter-Authority Phase 28 Result

Date: 2026-08-25  
Status: `PASS_THETA_MEASURE_PILOT`

## Receipt

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot/`

The pilot generated fresh stateless particles from a geometry-informed proposal
warm start. It did not load or reuse the parent replay bank. Target and
proposal terms are both explicitly in the `theta in R^4` measure; the stored
protocol says `density_jacobian=none_in_theta_log_terms`.

| Arm | Role | Target/status | ESS fraction | Weighted negative fraction |
|---|---|---:|---:|---:|
| C0 | fresh theta descriptive comparator | 64/64 | `0.9494` | `0.5007` |
| M0 | fresh theta candidate, not SMC-U admitted | 64/64 | `0.9796` | `0.7125` |

Both arms reached beta one and passed finite density, protocol, and ancestry
gates. The mass and mode values are descriptive; the proposal laws differ and
there is one seed, so no ranking is supported.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Nominate fresh M0 theta bank for a role-limited ETPF check | all paired finite/status/measure gates pass | no hard veto | no SMC-U proof, one seed, proposal/mode sensitivity | apply source-faithful ETPF directly to `[N,4]` rows | no authority, posterior, IID whitening, mode theorem, LEDH, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for C0 and M0 |
| Statistically supported ranking | none |
| Descriptive-only differences | ESS, mass, mode occupancy, and root counts |
| Default-readiness | not ready |
| Next evidence needed | fresh-bank ETPF role receipt and independent replication if promoted |

## Red-team note

The high M0 negative occupancy could reflect proposal calibration, finite
resampling, or a genuine target asymmetry. It is not evidence of mode discovery
or superiority. A target-level ETPF receipt can only test the transform's role
and finite/status compatibility.

