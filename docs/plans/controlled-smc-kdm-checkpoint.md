# Controlled-SMC / KDM active checkpoint

Updated 19 October 2026. The requested supervised-guidance implementation
plan is complete through its bounded T2/T5 pilot. Active scientific question:
can cSMC/KDM-inspired guidance improve the full analytical LEDH score when
selected using simulation labels? OT remains epsilon 1, five iterations;
32 terminal balancing updates also remain fixed. No OT tuning.

Worktree: /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922
Branch: ledh-refactor-with-policy-fix
HEAD: ea1018de1da55ed02a29a8d60f08a6827fce8a42
Preserve unrelated dirty work. Owner now requests a commit and a thorough
review of why the proposal has not established score improvement. No push
or branch synchronization requested.

Checked: four new propositions/proofs; exact-law SIR labels; optional Gaussian
product-moment and admissible indefinite-quadratic consumers; actual shared
UKF/flow/Contract-E/GenUT analytical recursion. Thirty-four distinct CPU tests
pass. T2 pilot-02 has 80/96 valid rows; raw-guide failures were a legacy PSD
check mismatch, repaired before fresh T5 pilot-03. T5 has 96/96 valid rows and
six passing finite-program directional checks. No GPU job remains.

T5 unguided validation loss is lowest. Guided test means are lower but are
dominated by one sequence and noise-score errors; every guide loses on another
sequence. Four test datasets establish no statistical ranking. No guide,
default, canonical tuning or HMC promotion. This is finite-bank selection,
not neural KDM retraining or continuous coefficient optimization.

Current plan: ledh-supervised-guidance-20261019.md
Result and exact evidence: artifacts/ledh-supervised-guidance-20261019/result.md
Terminal audit: artifacts/ledh-supervised-guidance-20261019/terminal-audit.json
Math/code audit: artifacts/ledh-supervised-guidance-20261019/audit.md
Manuscript: ../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex
New section: ../papers/ledh_younis_kdm_score/ledh_supervised_guidance.tex
Final PDF: ../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf
86 pages; section 15, pages 75--80, contains the new proposal and pilot results.
Two clean pdflatex passes; title and pages 75--80 visually inspected.

MathDevMCP proves two scalar algebra identities; document/probability checks
failed or were unencodable. Claude review could not authenticate. Direct
proof review and executable checks are recorded; external full certification
is not claimed. Build with installed pdflatex (latexmk absent).

Owner budget: 43200 GPU-process seconds; prior charge 9325.730914790, this
stage 570.83 (including conservative 60-second initialization reserve).
Total 9896.560914790; remaining 33303.439085210. CPU stage reserve 150 seconds
within 600. All failed attempts and prior stage evidence remain preserved.

Next research action: write a bounded fresh SIR scope-specific calibration
and independent-dataset protocol. First distinguish poorly calibrated
finite-program derivatives and label variance from guide benefit; include
unguided retention, frozen independent heuristic comparisons, numerical-control
justification, adequate dataset replication and parameter coverage. Preserve
OT (1,5). No direction-rejection or continuation veto fired; the completed
pilot does not itself authorize an unplanned default change.

Earlier diagnostics and budget details are linked in:
artifacts/ledh-sir-guidance-20261019/result.md
artifacts/ledh-sir-beta-repair-20261019/result.md
