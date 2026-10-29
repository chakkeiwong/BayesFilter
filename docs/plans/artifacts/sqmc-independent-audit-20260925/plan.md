# SQMC independent audit — 2026-09-25

Question: Does the previous agent's SQMC work compute its claimed mathematical targets, and do implementation and artifacts support the campaign conclusions?

Scope: transfer campaign commits through e6e5fc40, their shared SQMC/canonical-score call chains, tuning/oracle/route-comparison consumers, and integration changes through f3995a06. This is a correctness audit; no production promotion, tuning, benchmark ladder, or repair campaign is authorized by this plan.

Pre-execution skeptical audit: closeout identifies substituted targets (finite values in place of scores), missing scope tuning, and missing execution provenance. These are audit hypotheses to verify, not baselines to adopt. Use exact callback derivatives, known Hilbert fixtures, same-finite-program finite differences/autodiff, and checked paper equations as references. Existing passing tests do not prove model fidelity or production validity. Preserve the checkout and historical results; write only audit notes and diagnostic probes.

Evidence contract: primary pass criterion is equality with the stated target on independently derived deterministic fixtures and correct wiring from consumer to implementation. Missing derivative terms, target mismatch, invalid-state acceptance, scope mismatch, or artifact/claim contradictions veto the associated claim. Timing and stochastic value differences are explanatory only. No ranking, asymptotic accuracy, HMC readiness, or GPU/XLA readiness follows from this audit's CPU probes. Preserve commands, logs, source anchors and verdicts here.

Execution: conda tftwogpu; all probes deliberately CPU-only (CUDA_VISIBLE_DEVICES=-1, BAYESFILTER_TEST_DEVICE_SCOPE=cpu). No GPU initialization/probes, package changes, external messages, or expensive research experiments. Budget: at most 30 minutes CPU test/probe wall time, with each selected batch bounded; stop a reproducer once the defect is established. Static/source inspection continues independently. No subagents.

Checks: (1) primary-source equations and custom-target derivation; (2) data/model/oracle/parameter and derivative parity; (3) RQMC generation, ordering, ancestry and weights; (4) correction/reset/validity and loop merge; (5) tuning scope and four-route comparability; (6) artifact counts/provenance/inference; (7) focused executable regressions and adversarial fixtures; (8) severity-ranked final report with checked/unchecked boundaries.
