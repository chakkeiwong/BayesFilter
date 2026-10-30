# Surrogate-HMC component campaign: stopped at authorized budget

Updated 2026-10-09T01:26:07.398593+00:00. Checkout /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917; branch ledh-graceful-failure-execution-20260917; HEAD f5e69d716818ff9ecf28b732a8952f5ec7f87739. Preserve unrelated staged HMC changes.

Question: does marginal-mixture weighting preserve LGSSM/KSC likelihood and analytical-score accuracy and reduce range-bearing errors on fresh paired data? Ultimate goal: testing surrogate HMC. This phase tests likelihoods/scores only.

Status: INCOMPLETE. The extended master exited normally at 2026-10-08T08:25:20.809719+00:00; no campaign worker is live. Charged worker time172800.219s against authorized172800s (48hours); remaining compute budget0. Do not launch or auto-resume numerical experiments without a renewed budget.
Coverage:27/90 jobs complete (17current-source,10retained),9partial,1started without completed comparison,53not started. N1024:22complete/8partial; N4096:5complete/1partial/1unfinished/23pending; N16384:30pending. Partial rows remain resumable.
Terminal review: terminal-review-20261009.md. Detailed frozen results: results.md and analysis.json. Master state: campaign.json and next-phase.json. Extension history/command: budget-extensions/20261007-01/status.json.

Findings: completed LGSSM cells all meet value/score noninferiority. Six KSC cells remain inconclusive. Two M13 cells have invalid ancestor baselines. Total77parameter-point comparisons:69noninferior,6inconclusive,2invalid baseline. Supported lower-error claims are per cell relative to ancestor weighting, not universal. Simple-comparator promotion vetoes remain. No default or HMC-readiness conclusion.
Latest completed job: LG10 T10 N4096,16datasets, all3points supported lower value/score errors. Worker and campaign timeouts leave longer scopes incomplete; they do not demonstrate numerical invalidity or reject the research direction.
Engineering and N4096 precision checks passed. Terminal integrity check matched all837source hashes, found all27completed-result files, and found no live campaign workers. Evidence: terminal-integrity-20261009.json. Twelve continuation-controller checks passed earlier. Executed source is archived in executed-source.tar.gz; preserve it and all prior evidence.
Selective FP64 covariance-product repair passed68same-process public-consumer checks, preserving healthy rows and recovering12invalid high-dimensional rows. Cross-process FP32/TF32 repeatability remains unresolved; see detailed report.
LaTeX monograph/proposal were built and visually reviewed under ../surrogate-hmc-marginal-expanded-20261006-03/documentation/. Non-native KSC/range-bearing dimension variants still require model definitions.

Plan: docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md; extension: docs/plans/surrogate-hmc-marginal-budget-extension-2026-10-07.md.
Exact next action: report the budget-limited terminal result. A later budgeted resumption of unfinished jobs belongs to this same master plan. Preserve completed scopes and partial rows, consumed time, frozen settings, source identity and scientific vetoes. No new direction or additional compute has been authorized by a status question.
