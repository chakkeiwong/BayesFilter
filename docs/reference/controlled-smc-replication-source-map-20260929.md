# Controlled-SMC replication source map

Checked 29 September 2026. This map separates the executable source demo from
the empirical results it cannot establish. The active master is
[controlled-smc-kdm-replication-master-program-20260928.md](../plans/controlled-smc-kdm-replication-master-program-20260928.md).

The publisher version was recovered from the
[Oxford archive](https://ora.ox.ac.uk/objects/uuid%3A4f4efb19-4344-4168-b762-3808d6fadfc2/download_file?safe_filename=Heng_et_al_Controlled_sequential_Monte_Carlo.pdf&type_of_work=Journal+article).
It is *Annals of Statistics* 48(5), 2904–2929, DOI 10.1214/19-AOS1914.
Local PDF: `.localresources/papers/heng2020-controlled-smc-published.pdf`;
SHA-256: `573c9bcc09eb6baf0bc48da50676f2baae9ffc0575e469af143d6b193250e9eb`.
A searchable text copy is beside it. The previously inspected arXiv v3 and its
appendices remain in `heng2020-controlled-smc.pdf` and `.txt`.

For the selected work, the published optimal recursion (19), least-squares
equations (20)–(21), refinement Algorithm 3 and neuroscience settings match the
inspected manuscript. Proposition 1 becomes Proposition 4.1. This is a scoped
version check, not a complete equivalence audit of both articles/supplements.
Method anchors in the published text: lines 498–721; neuroscience: 1088–1123.

All source entries below are under
`.localresources/code/heng-controlled-smc/StateSpaceModels/NeuroScience`,
author commit `2696f6d611aae172d10102cb7bfbfd314bb6b88f`.

| Cell | Source and paper setting | Current evidence |
|---|---|---|
| Demo / numerical controls | `Demo.m`: N=1024, I=2, all 3000 observations | Original author execution and scalar TF GPU/XLA execution pass; not an empirical replication |
| Figure 4, right: likelihood variance | Published §6.1; `NormConst_cSMC.m:33`, `NormConst_BPF.m:33`: alpha=.99, variance .01:.01:.20, I=3, cSMC N=128, BPF N=5529, 100 repetitions each | Full source matrix running; live status in the master checkpoint |
| Figure 4, left: smoothing | `Smoothing_cSMC.m:45`; N=1024, I=3, FFBS comparison | Separate smoother claim; no new smoothing replication |
| Figure 5: PMMH | §6.1; 100,000 transitions | Outside the initial filtering/normalizer work; not executed |
| Lorenz-96 §6.2 | Separate multidimensional source/model and experimental matrix | Source matrix mapped in the remaining-source audit; full-covariance RK4/APF implementation and normalization checks remain open |
| ToyMultimodal | Manuscript Appendix F, equation (78), Figure 9/Table 2; I.1 algebra | NNLS/RBF method mapped; current quadratic core does not implement it; missing Repeat driver and 100/101-observation discrepancy remain |

The first empirical filtering target is Figure 4 right. The source statistic is
`log10(var(logZ, ddof=1)/mean(logZ)^2)`, implemented in
`NormConst_Results.m:13`. It is not the variance of the likelihood estimator.
Source particle counts remain a fixed-settings replication arm; their original
CPU cost matching does not establish equal cost on our GPU implementation.
Preserve full backward-fitting/adaptation costs when constructing a separately
matched-compute comparison. Do not select a setting on the final variance grid.

The Figure 4 source matrix now has a driver, predeclared uncertainty and
a completed feasibility/parity pilot. Other experiments still require their own
executable specification and checks. BPF, observation-guided EKF/UKF proposals and any applicable
source APF form the master's wider heuristic checks; FFBS is required only for
the separate smoothing claim. A figure reproduction and a claim of superiority
over those extra baselines are distinct decisions.

The current scalar consumer does not implement Lorenz dynamics, full covariance
policies or a general multivariate adapter. Its normal-equation solve and .001
scalar precision projection reproduce explicit source conventions. Passing a
healthy scalar fixture does not validate full-quadratic projection or select a
universal numerical safeguard. Original code is executed externally; the TF
implementation is independently derived from the mathematical identities.

The [remaining-source audit](controlled-smc-remaining-source-audit-20260929.md)
records the Lorenz normalization/data issues and the newly located multimodal
supplement with its different RBF fitting procedure.
