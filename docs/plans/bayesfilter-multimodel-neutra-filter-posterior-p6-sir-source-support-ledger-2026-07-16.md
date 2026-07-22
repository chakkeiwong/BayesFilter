# P6 SIR Zhao-Cui Source-Support Ledger

Date: 2026-07-16

Decision: `SOURCE_BOUNDARY_CHECKED_TARGET_EXTENSION_REMAINS_BLOCKED`

## Minimal Scholarly Audit

| Field | Result |
| --- | --- |
| Metadata date | 2026-07-16 |
| Seed paper | Zhao and Cui, *Tensor-Train Methods for Sequential State and Parameter Learning in State-Space Models*, JMLR 25 (2024), 1-51 |
| Classification | `DIRECT_METHOD` for TT/SIRT sequential state/parameter learning; Section 6.3 is an `EMPIRICAL_EXAMPLE` with fixed SIR parameters |
| Local full text | `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`, SHA-256 `c547b9af2e407c7a0d28bf49ca594fed65d9794d4f37ca605edebd91f9755e35` |
| Author source | `third_party/audit/tensor-ssm-paper-demo`, upstream snapshot already recorded by P0 |
| Publication/retraction | published JMLR text inspected; no notice in local full text; live retraction index not queried |
| Citation/venue metadata | not queried; irrelevant to target equality and not truth evidence |
| Forward snowball | not queried; no local citing-work index and not needed to decide the present source mismatch |

## Primary Technical Anchors

- Paper equations (9)-(12), Proposition 2, and Algorithms 1-2 define the
  sequential joint approximation, marginalization, squared-TT density, and KR
  operations.
- Paper Algorithm 5 defines preconditioned replacements.
- Paper Section 6.3 equation (37) fixes `kappa_j=0.1` and `nu_j=18`, states
  explicitly that the task infers `S_j,I_j,R_j`, uses J=9, T=20, observations
  at `k=1,...,T`, and an 18-dimensional state `(S1,I1,...,S9,I9)`.
- Author `eg3_sir/mainscript.m:12-18` sets `d=0`, `m=18`, `n=9`, `T=20`.
- Author `models/full_sol.m:21-129` pushes samples, recomputes weighted affine
  coordinates, builds/retains `TTSIRT`, marginalizes the retained object, and
  evaluates its density.
- Author `models/pre_sol.m` implements the corresponding preconditioned route.

## Claim Support

| Claim | Support class | Verdict |
| --- | --- | --- |
| Zhao-Cui SIR estimates states with fixed rates | `PRIMARY_TECHNICAL_SUPPORT` | correct |
| BayesFilter's three log-scale SIR posterior reproduces Zhao-Cui Section 6.3 | paper/source comparison | wrong relative to that claim |
| The three-parameter target may be studied as a BayesFilter extension | project target definition | allowed if labeled extension |
| Existing local complete-data score is a full observed-data score | code/result inspection | wrong relative to that claim |
| Current fixed-TTSIRT substrate is HMC-score ready | explicit local blockers | unsupported and currently blocked |

## Snowball And Omission Status

P0 already recorded the backward-snowball anchors for KR/TTSIRT mathematics,
including Cui-Dolgov and Spantini et al. They are relevant before implementing
new transport mathematics, but not needed to decide the present direct mismatch
between `d=0` author SIR and `d=3` BayesFilter inference. Forward snowballing,
live retraction metadata, and recent replication searches remain publication-
grade omission risks, not blockers for this conservative target classification.

## What Is Not Concluded

No Zhao-Cui observed-data parameter posterior, full retained-marginal score,
HMC readiness, posterior correctness, approximation accuracy, scalability,
superiority, or production readiness is established.

