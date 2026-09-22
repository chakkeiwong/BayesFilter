# Fresh research conversation: observation-aware TT proposals

The owner's unresolved question is: derive a correct way to incorporate the
current observation into Zhao–Cui TT regression/proposals, assess whether UKF
guidance is a good choice, and survey the literature for alternatives. This is
a literature and mathematical-design task. Do not substitute the separate
auxiliary-particle-filter campaign for it.

Start from this brief in a fresh conversation. Do not load or fork the failed
session, incident investigation, pasted conversation, or large recovery history.
Follow the applicable repository instructions and scholarly literature audit
skill; keep its source requirements while reading only the needed ranges.

Research checkout: `/home/chakwong/BayesFilterZhaoCui`, observed branch
`zhao-cui-tt-regression-20260908`. Notes checkout:
`/home/chakwong/BayesFilter`, observed branch `surrogate-hmc`. Both have dirty
work; verify current status and preserve unrelated changes before editing.

Existing leads, whose scientific claims require verification rather than
automatic adoption:

- `bayesfilter/highdim/zhao_cui_algorithm2_preparation_tf.py` in the research
  checkout contains the observation-weighted sigma-point helper and TT
  preparation entry point. First inspect their definitions and exact callers.
- The UKF-guided TT plan is
  `docs/plans/bayesfilter-zhao-cui-algorithm3-ukf-guided-tt-master-program-2026-09-09.md`
  in that checkout. Search headings first; it is about 23 KB.
- The active manuscript identified in the previous conversation is
  `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`
  in that checkout. It is about 153 KB; search relevant headings/equations and
  read selected ranges. The canonical chapter is
  `docs/chapters/ch19c_dpf_implementation_literature.tex`.
- The notes checkout has
  `docs/plans/artifacts/zhao-cui-integrated-audit-20260911-04/repair-proposal.md`
  and earlier stage results. Treat them as leads to their source/code anchors.

The last completed conversation reported observation-aware TT target fitting
but no end-to-end UKF chart integration. Its four unresolved audit issues were
rank activation, nonfinite aggregate acceptance, proposal sampling-density
mismatch, and missing claim-bearing consumer wiring. These statements were
not re-audited by the incident investigation. No new literature result was
produced in the failed final turn; it stopped after reading a skill file.

First bounded stage: identify the conditional target, regression measure,
coordinate/chart role, sampling law, and importance-weight denominator from
the cited author paper/code and exact local call chain. Then frame the source
search around the actual unresolved observation-incorporation problem.
Distinguish source-faithful operations, fixed adaptations, and local extensions;
do not presume UKF is optimal or a sigma-point moment helper is a complete UKF.

Save a concise stage result and next action. Bound each combined tool response
to about 4,000 tokens, normally 2,000 per command or less; read exact ranges,
extract selected structured fields, and save complete logs to disk. No GPU,
training, HMC, or benchmark campaign is needed to begin this literature stage.
