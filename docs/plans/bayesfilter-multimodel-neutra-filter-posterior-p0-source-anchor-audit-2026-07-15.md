# P0 Zhao-Cui Source Anchor Audit

Date: 2026-07-15

Decision: `PASS_FOR_CONSERVATIVE_ROUTE_CLASSIFICATION`

## Minimal Scholarly Audit

| Field | Result |
| --- | --- |
| Metadata date | 2026-07-15 |
| Seed paper | Zhao and Cui, *Tensor-Train Methods for Sequential State and Parameter Learning in State-Space Models*, JMLR 25 (2024), 1-51 |
| Local full text | `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`, SHA-256 `c547b9af2e407c7a0d28bf49ca594fed65d9794d4f37ca605edebd91f9755e35` |
| Author source | `third_party/audit/tensor-ssm-paper-demo`, upstream commit `80034dccb99eb1d86284a1839b4a12067d13b9da` |
| Publication status | Published JMLR full text; front matter states CC-BY 4.0 |
| Retraction status | No notice in inspected local published full text; live retraction index not queried |
| Citation/venue metadata | Not queried; not required for implementation identity and not truth evidence |
| Forward snowball | Not queried; no trustworthy local citing-work index |
| Quarantined source | Cached `openalex_zhao_cui_jmlr_2024.json` is mislabeled and contains unrelated computer-vision records |

Machine-readable scholarly ledgers are under the attempt-04 artifact root:
`source_support.json`, `citation_venue_metadata.json`,
`backward_snowball.json`, `forward_snowball.json`, `claim_support.json`, and
`omitted_paper_risks.json`.

## Inspected Primary Anchors

- Paper equations (9)-(12) and Algorithm 1 define the recursive joint
  `(theta, x_t, x_(t-1))` approximation and marginalization of `x_(t-1)`.
- Proposition 2 and Algorithms 2-4 define squared-TT marginal densities,
  conditional KR maps, and sequential particle/path algorithms.
- Section 5 and Algorithm 5 define preconditioned reapproximation and retained
  marginal/conditional maps.
- Section 6.2 estimates SV parameters `(gamma,beta)` with fixed `sigma=1`.
- Section 6.3 fixes SIR `kappa_j=0.1` and `nu_j=18` and estimates states only.
- Section 6.4 estimates six predator-prey parameters `(r,K,a,s,u,v)`.
- Author `models/full_sol.m:21-129` pushes samples, adaptively reapproximates,
  recenters, constructs `TTSIRT`, retains `SIRTs`, and updates normalizers.
- Author `models/pre_sol.m:16-213` implements the preconditioned counterpart.
- Author example scripts set `d=2` for SV, `d=0` for SIR, and `d=6` for
  predator-prey.

## Route Classifications

| Operation/route | Classification | Reason |
| --- | --- | --- |
| Joint transition/likelihood assembly, marginalization, retained SIRT concept | `source_faithful` only for the specifically anchored operation | Direct paper and author-source match |
| Frozen samples, ranks, bases, schedules, branch choices for differentiable HMC | `fixed_hmc_adaptation` | Preserves an author operation while freezing discrete/random choices; still requires operation-level proof |
| Current `SVX-ZC` factorized scalar fixed-design wrapper | `extension_or_invention` as a whole route | Wrapper disclaims coupled/adaptive TTSIRT and delegates to the local fixed fitter |
| Current `PP-ZC` generic all-axes retained grid | `extension_or_invention` and production-ineligible | It retains/propagates a full grid instead of the author SIRT retained object |
| Parameterized `SIR-ZC` | Extension target with incomplete `fixed_hmc_adaptation` substrate | Paper SIR has `d=0`; BayesFilter infers three scale parameters |
| Chapter 18b `STR-ZC` | `extension_or_invention` by definition | Neither paper nor author examples contain this structural model |

## Backward Snowball And Omissions

The checked paper points to Cui-Dolgov (2022) for the squared inverse Rosenblatt
transport and Proposition-2 foundations, Spantini et al. (2018) for decomposition
structure, Griebel-Harbrecht (2023) for approximation-error analysis, and Reich
(2013)/Spantini et al. (2022) as transport competitors. These sources are not
needed to conservatively reject current route-wide faithfulness, but Cui-Dolgov
must be inspected before implementing or promoting new KR/TTSIRT mathematics.
Recent citing works and replications remain a publication-grade omission risk.

## What Is Not Concluded

No current BayesFilter Zhao-Cui posterior is admitted; no source-faithful full
route, approximation accuracy, HMC readiness, posterior correctness,
superiority, or production readiness is established. The audit only supports
the conservative P0 classifications and future source-gate requirements.

