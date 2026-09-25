# Mixture-UKF Literature Audit

metadata_date: 2026-09-02
scope: observation-informed proposals and recursive non-Gaussian filtering

## Decision

The established methods closest to the proposed design are the Sigma-Point
Particle Filter (SPPF, also called the Unscented Particle Filter or UPF) and
the Gaussian-Mixture Sigma-Point Particle Filter (GMSPPF). A Gaussian-sum or
Gaussian-mixture UKF is the deterministic filtering counterpart. These are
different from a higher-order unscented transform or CUT4: the latter improve
moment integration for one Gaussian, while the former retain several Gaussian
components or several local conditional proposals.

The recommended generic hypothesis is a fixed-topology, per-ancestor UKF/APF
proposal with exact importance correction, tested first with one component and
then with two or four components. The GMSPPF recursion is a lower-cost global
mixture alternative, but its published weight formula uses an approximate
predictive mixture and therefore is not automatically an exact correction for
the BayesFilter target.

## Source-Support Ledger

| Source | Class | Technical material inspected | Supports | Does not support |
|---|---|---|---|---|
| R. van der Merwe, *Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models* (local PDF) | DIRECT_METHOD | Sections 5.1-5.3; Eqs. (31), (35), (36)-(42); Algorithm 1; Table 1 | Per-particle SPKF Gaussian proposals conditioned on the current observation; finite GMM/SPKF bank; EM/WEM reduction; source-specific cost comparison | Exactness or superiority for the current C2 model; correctness of the vendored implementation |
| van der Merwe, de Freitas, Doucet, Wan, *The Unscented Particle Filter* (2000/2001 lineage) | DIRECT_METHOD | Identified through the checked thesis bibliography; original report not separately obtained | Method lineage and SPPF name | Detailed source claims beyond the checked thesis reproduction |
| van der Merwe and Wan, *Gaussian Mixture Sigma-Point Particle Filters...* (ICASSP 2003) | DIRECT_METHOD | Identified in the checked thesis bibliography and Crossref metadata | Published GMSPPF method identity; DOI `10.1109/ICASSP.2003.1201778` | Current-target correctness or generic guarantees |
| Alspach and Sorenson, *Nonlinear Bayesian Estimation Using Gaussian Sum Approximations* (1972) | FOUNDATIONAL | DOI and citation identified; full text not locally obtained | Gaussian-sum lineage | Detailed theorem or algorithm claims |
| Pitt and Shephard, *Filtering via Simulation: Auxiliary Particle Filters* (1999) | FOUNDATIONAL | Local SMC notes cite the APF mechanism; full primary paper not separately obtained in this pass | Observation-based ancestor lookahead as the APF framework | UKF-specific proposal behavior |
| Easley and Berry, *A Higher Order Unscented Transform* (2021) | FOUNDATIONAL | Local PDF, Section 4, Definition 4.1, Theorem 4.2, Algorithm 4.1, Remark 4.4 | HOUT matches higher moments with a point rule whose rank/tolerance cost can grow | A multimodal density representation or an exact particle proposal |
| Ebeigbe et al., *Generalized Unscented Transformation* (2021/2025) | DIRECT_METHOD | Local PDF, abstract and Sections II-III | GenUT retains `2n+1` points and captures selected diagonal skewness/kurtosis | A mixture posterior or a guarantee against proposal-weight collapse |
| BayesFilter CUT4 chapter/code | PROJECT_DERIVATION | `docs/chapters/ch16_sigma_point_filters.tex`; `bayesfilter/nonlinear/cut_tf.py` | Current CUT4-G count `2n+2^n` and its one-Gaussian moment-rule boundary | A generic solution to multimodal proposal overlap |

Local primary artifact:
`docs/Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models Merwe(03).pdf`.

## Citation And Venue Metadata

DOI links checked or identified:

- [GMSPPF, van der Merwe and Wan](https://doi.org/10.1109/ICASSP.2003.1201778)
- [Gaussian-sum filtering, Alspach and Sorenson](https://doi.org/10.1016/0005-1098(71)90097-5)
- [Auxiliary particle filters, Pitt and Shephard](https://doi.org/10.1111/1467-9868.00280)
- [HOUT, Easley and Berry](https://doi.org/10.1137/20M135546X)
- [GenUT, Ebeigbe et al.](https://arxiv.org/abs/2104.01958)

Citation counts and venue rankings were not used as correctness evidence and
were not refreshed for this note.

## Snowballing And Omission Status

Backward snowballing from the checked thesis covers Gaussian-sum filtering,
SPPF/UPF, APF, EM/WEM mixture reduction, and sigma-point alternatives. The
web search endpoint returned temporary 502 errors during this pass, so a
forward citation search and a complete recent-literature sweep were not
possible. The original UPF report and the Alspach-Sorenson full paper remain
source-closure gaps for publication-grade historical claims.

## Claim-Support Map

- `PRIMARY_TECHNICAL_SUPPORT`: a UKF/SPKF can be run separately for each
  particle and conditioned on the current observation (checked thesis Sec. 5.2).
- `PRIMARY_TECHNICAL_SUPPORT`: a finite GMM can be propagated by a bank of
  SPKFs and compressed by EM/WEM (checked thesis Sec. 5.3).
- `PROJECT_DERIVATION`: for an exact BayesFilter target, the numerator must use
  the exact transition and observation densities; the UKF/GMM is proposal-only.
- `IMPLEMENTATION_EVIDENCE`: the current generic C2 branch has a global TT
  channel and a parent-conditioned defensive channel, while the LEDH Algorithm
  1 route already contains a per-particle UKF lifecycle.

## What Is Not Concluded

The literature establishes the method families, not that a two- or four-
component proposal will solve C2. It also does not establish that the current
vendor UPF code is a correct importance sampler; the repository review notes
that its displayed weighting path omits the transition/proposal correction.

## Next Required Evidence

1. Verify linear-Gaussian conditional parity with a fixed-branch implementation.
2. Compare K=1, K=2, and K=4 per-ancestor UKF/APF proposals using complete
   mixture densities and exact target ratios.
3. Add a fixed small Student-t defensive channel and report conditional ESS,
   weight tails, log-normalizer error, cost per effective sample, and gradient
   parity across independent seeds.
4. Only then evaluate a clustered/global GMSPPF-style recursion as a cost arm.
