# Gap-closure checkpoint — 10 October 2026

Active question: determine whether unresolved cSMC regression semantics and KDM
float32 coupled-training drift prevent the agreed author-relative replication.

Completed this stage:
- revised and skeptically audited the gap-closure plan;
- installed four literal project-scoped diagnostic prefixes;
- executed localization-01 with GPU preflight, TF32 off, XLA and memory growth;
- cSMC source controls pass; first divergence is fitted regression;
- KDM source coupled replay 64/64 controls pass; FP64 1808/1808 checks pass;
- KDM FP32 failures first appear in gradients and accumulate;
- charged 12.7435 CPU and 126.6963 GPU seconds; active campaign balance now reports 220871.59 CPU and 118130.81 GPU seconds.

Evidence:
- gap-closure-result-20261010.md;
- gap-closure-20261010/localization-01/manifest.json and decision.json;
- csmc/summary.json and kdm/comparison.json.

Remaining:
- explain or bound the cSMC conditional solver and retracing warning;
- decide whether a prospective cSMC precision design is feasible;
- measure KDM downstream non-harm or write a concrete three-unit K1 plan;
- do not call either replication complete from localization diagnostics.

Active continuation: docs/plans/controlled-smc-kdm-root-causes-20261010.md.
Skeptical audit and syntax checks passed. Hypotheses: conditional Octave solver,
shared angular conditioning, and Adam scalar arithmetic. No runtime repair yet.
Next action: --phase root-causes, output root-causes-20261010/root-causes-01.
Reserve 600 CPU / 1800 GPU seconds, two attempts; no qualification extension.
