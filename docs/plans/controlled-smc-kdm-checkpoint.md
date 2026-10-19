# Controlled-SMC / KDM active checkpoint

Updated 20 October 2026. Supervised-guidance pilot and requested failure review
are complete. Active scientific question: can correctly composed,
downstream-supervised cSMC/KDM-inspired guidance improve the full analytical
LEDH score? OT remains epsilon 1, five iterations, and 32 terminal balancing
updates. No OT tuning or new GPU campaign in this review.

Worktree: /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922
Branch: ledh-refactor-with-policy-fix
Implementation/pilot commit: 4abbaf9f1; review committed separately.
Preserve unrelated dirty work. Owner requested commit and thorough review.
No push or branch synchronization requested.

Checked: supervised-risk identity correct under its assumptions; actual
shared UKF/flow/Contract-E/GenUT analytical consumer passes scoped tests.
Guide fitted on physical states is applied before the flow, producing
h(F^{-1}(x)) rather than h(x) afterwards. New proposition and exact Gaussian
counterexample establish the distinction; four focused consumer/density/
Jacobian/derivative checks pass. This is a design limitation, not an incorrect
importance correction. Its causal SIR effect is not yet measured.
T5 retains 96/96 valid rows, six directional checks and ten matching runtime
hashes. Earlier T2 raw-guide PSD failures were repaired and retained separately.

T5 unguided validation loss is lowest. Guided test means are lower; every guide
loses on sequence 261029508. Over 99.3% of scaled loss is in the noise coordinate.
Four test datasets establish no statistical ranking. No promotion. This is
finite-bank selection, not neural KDM retraining or coefficient optimization
against score risk. OT/reset causality remains unmeasured.

Current review: ledh-supervised-guidance-failure-review-20261019.md
Findings/commands: artifacts/ledh-supervised-guidance-20261019/failure-review/review.md
Exact arithmetic: same directory, analysis.json and analyze.py
Implementation plan: ledh-supervised-guidance-20261019.md
Result and exact evidence: artifacts/ledh-supervised-guidance-20261019/result.md
Terminal audit: artifacts/ledh-supervised-guidance-20261019/terminal-audit.json
Math/code audit: artifacts/ledh-supervised-guidance-20261019/audit.md
Manuscript: ../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex
New section: ../papers/ledh_younis_kdm_score/ledh_supervised_guidance.tex
Final PDF: ../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf
88 pages; new flow-coordinate derivation is subsection 15.7, pages 80--81.
Two clean pdflatex passes; rendered pages 80--82 visually inspected.
Prior 86-page PDF protected in /tmp.

MathDevMCP proves the scalar pullback and prior algebra identities; document/probability checks
failed or were unencodable. Claude review could not authenticate. Direct
proof review and executable checks are recorded; external full certification
is not claimed. Build with installed pdflatex (latexmk absent).

Owner budget: 43200 GPU-process seconds; prior charge 9325.730914790, this
stage 570.83 (including conservative 60-second initialization reserve).
Total 9896.560914790; remaining 33303.439085210, unchanged by this review.
Review CPU cap 600 seconds, GPU zero; review-run.json records checks and
attempts. One wrong-environment collection failed; tftwogpu retry passed.
No environment was changed. All earlier evidence remains preserved.

Next research action: plan/test an optional flow-aligned quadratic guide using
the actual shared affine map and its total analytical derivatives; then a fresh
bounded original/zero/aligned comparison with downstream supervised calibration,
fresh datasets and particle streams, and saved per-step diagnostics. Preserve
OT (1,5). The review does not launch this campaign or select on old test data.

Earlier diagnostics and budget details are linked in:
artifacts/ledh-sir-guidance-20261019/result.md
artifacts/ledh-sir-beta-repair-20261019/result.md
