# M13 particle-resolution result — 2026-09-24

The N4096 canonical force completes all nine nominal T20 calibration cases in
FP32/TF32/XLA. N1024 completes eight; N256 completed seven. This nominates N4096
for fresh scope-specific tuning, not posterior or default admission. The initial
particle designs are not nested, so the ladder does not establish a statistical
ranking or isolate particle count from realized design variability.

| N | Valid nominal rows | First two time0 ESS | Interpretation |
|---|---|---|---|
| 256 | 7/9 | 1,1 | Preserved failed calibration |
| 1024 | 8/9 | 1.002,1.009 | One typed correction failure remains |
| 4096 | 9/9 | 1.345,1.827 | Viable resolution hypothesis |

No unflagged failures occurred. All rows retain full canonical correction and
analytical score. The N4096 exact chunk policy uses K2048. Individual arm wall
times were44.39s and52.26s, including tracing/compilation; these are not
steady-state or HMC cost measurements.

Run02 completed after37 review tests. Its source snapshot and manifests preserve
all commands, environment, data, seeds, device/growth settings and hashes under
`artifacts/surrogate-hmc-closure-20260922-m13-force-recovery-02/`.
Run01 failed before filtering on manifest serialization; both attempts are
charged:325.885016CPU/109.983642789GPU seconds. Global cumulative
45675.573978292CPU/35398.270696728GPU; remaining8324.426021708CPU/191401.729303272GPU.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue master at N4096 | 9/9 nominal valid on saved calibration | No new numerical veto | Generalization to fresh designs/data and force stability | Fresh scoped calibration/validation | Posterior accuracy or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | N1024 still has one flagged failure; N4096 passes this screen |
| Statistically supported ranking | None |
| Descriptive differences | Validity counts, ESS and total arm wall time |
| Default readiness | Not established; FP32 policy unchanged |
| Next evidence needed | Independent scope tuning and posterior reference/endpoint |

Post-run red team: a fresh design may still collapse; the posterior endpoint
remains unqualified. Do not interpret force validity as accurate likelihood or
posterior evidence. LG1 is closed and is not rerun.
