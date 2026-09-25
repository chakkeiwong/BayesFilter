# M20 CPU development result

The declared smaller-epsilon grid did **not** repair centered-funnel tuning.
No pair survived independent verification under either the original frozen
geometry or a fresh preparation. The original target and acceptance policy
were preserved. The combined grid/refinement hypothesis is therefore not a
demonstrated repair and has not been promoted to an automatic default.

| Centered funnel | Original search | Expanded exploration |
| --- | ---: | ---: |
| Saved M17 geometry, new scopes and streams | 0 verified / 16 pairs | 0 / 100 |
| Fresh merged-source preparation | 0 verified / 15 pairs | 0 / 100 |

Preparation completed. Numerical rejection and valid high-acceptance evidence
coexist; invalid endpoints do not supply acceptance directions. The finite
candidate cap leaves untested settings, so an empty result does not prove that
every centered-coordinate pair fails. It does show that simply declaring a
broader small-step grid did not solve this case. Checked noncentered coordinates
remain a target-preserving diagnostic option, with their own posterior evidence.

The rotated Gaussian retained 20 verified members and the noncentered funnel
retained 19. Before posterior execution the worker recorded the first verified
ID at each of L=(3,9,18,25), preserving all siblings. All four rotated members
reached 10000 retained draws per chain without meeting every 0.05 absolute
mean/median MCSE target. Three noncentered members reached the same precision
cap; L=18 instead reached the 10000 warmup cap and produced no retained draws.
None of the eight members passed all posterior checks. There were no chunk
hard vetoes in those posterior runs.

These are clustered, one-fit development outcomes, not eight independent
replications and not a ranking of L. The Gaussian median planning calculation
and funnel child variance explain why absolute accuracy is costly; they do not
prove impossibility or authorize changing the requirement. Lugsail estimates
retained mean uncertainty; it does not certify burn-in. The official book and
agent reference now explain this distinction and the units of precision. The
570-page book was rebuilt, its bibliography resolved, and changed printed page
453 was inspected and installed in `docs/main.pdf`.

## Execution and continuation

The first four worker launches failed because the diagnostic harness omitted
the required acceptance-policy argument. The harness was repaired by supplying
the unchanged typed policy; every failed attempt is preserved and charged.
The completed CPU tranche consumed **1397.7616405547597 worker-seconds**. With
M19, the replacement allowance has **170709.93234966812 CPU and 86400 GPU
worker-seconds** remaining before M21. Source-bound old-geometry and merged-source
results remain distinct. No old checkpoint was relabeled as current-source
evidence.

Trusted readiness checks found no idle policy-permitted GPU. The GPU repetitions
remain pending; this is a scheduling constraint and provides no evidence of a
hardware or method failure. The independent M21 CPU controller study proceeds
under its resolved design. Its public Gaussian/beta-binomial controls also
complete the shared-path integration obligation. Further geometry and learned
transport work remain explicit in the master; this CPU tranche does not close
those scientific gaps.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
| --- | --- | --- | --- | --- | --- |
| Do not automate the epsilon-grid hypothesis | No verified centered pair | Invalid candidates remain rejected | Untested pairs and geometry suitability | Checked coordinate/geometry diagnosis; preserve this negative comparator | No universal impossibility claim |
| Preserve all 39 verified siblings | Identity and verification intact | Eight assessed posteriors miss required checks | Member dependence, MCSE calibration and sample cost | M21 actual stopping study and independent controls | No posterior promotion |
| Continue funded CPU work | Valid completed records and budget | GPU capacity unavailable | GPU transfer and fresh confirmation | M21; revisit GPU scheduling | No GPU closure |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Invalid tuning pairs stay rejected; warmup and precision caps remain visible |
| Statistically supported ranking | None |
| Descriptive differences | Candidate counts, member MCSE and stop counts |
| Default readiness | Unchanged |
| Next evidence | Independent controller calibration, actual HMC controls, target-specific geometry/transport repair and GPU repetitions |

Post-run review: the strongest alternative explanation is insufficient search
coverage under an unsuitable frozen geometry, rather than a missing epsilon
repair. A fresh independently verified centered pair would overturn the narrow
usability result. Heavy-tailed functional uncertainty and single-fit member
selection are the weakest posterior evidence. Repeating the same matrix until
a favorable member appears would not resolve those limitations.

Compact evidence and exact attempts are in
`artifacts/hmc-repair-master-2026-09-16/m20-r1/cpu-evidence-audit.json` and
`reconciliation-cpu-tranche.json`; raw tensors, receipts and immutable snapshots
remain in their versioned local directories.
