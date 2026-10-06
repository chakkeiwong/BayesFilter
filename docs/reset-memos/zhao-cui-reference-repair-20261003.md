# Zhao--Cui reference repair checkpoint

Active request: plan, review, and execute original-code reference repairs.
Branch: `sqmc-development`; base commit `a925f67a`. Engineering execution and final validation complete;
this checkpoint is included in the task commit. No merge/push in this plan.
Plan: `docs/plans/zhao-cui-reference-repair-20261003.md`.
Result: `docs/benchmarks/zhao-cui-reference-repair-results-20261003.md`.
Evidence root: `docs/plans/artifacts/zhao-cui-reference-repair-20261003/`.

Completed: paper/source audit; immutable extraction of 19 callbacks; Octave
compatibility overlays; stable logmeanexp with all raw weights retained; stable
Gaussian smoothing logs; classified fixed-target PP adapter; exact saved-data
hash verification; executed source call chain and five-state model parity.
Final author run: `author-reference-02`; final T=20 run: `fixed-target-02`.
Source fingerprints match before/after. Failed attempts remain in the ledger.
Octave budget consumed 89.716/900 seconds; no more research runs planned here.

Checked values: PP -97.335236612, ESS 51.379/64; SIR -4253.054415329, ESS 1/64.
Saved approximate bootstrap references: -97.342973851 and -678.077466314.
One source seed supports no statistical ranking. SIR is unsuitable as an
accuracy reference at these tiny settings. PP is a diagnostic comparator only.
Saved PP float32 parameters and source float64 point differ by up to 2.384186e-8;
see comparison-parameter-audit.json. No analytical score is implemented.

Final validation, 2026-10-04: 11 focused regressions, Python compilation,
extraction-only/source-immutability smoke and rejected-preparation record check
pass. Evidence is in `validation/validation.json` and its individual logs.

Source recheck after the user's challenge, 2026-10-04: published reproduction
has NOT been performed. Paper SIR uses ranks 10/20/40 and five ALS iterations;
our rank-4/one-pass/64-draw result cannot judge that method. The PP paper's
successful nonlinear pre_sol route was not executed. The disputed optional
fifth lml output is not requested by either author example driver. See the
new published-comparison section in the result note for exact anchors.

Next scientific question: reproduce the paper's ESS and trajectory findings
with matched author targets and declared paper/driver settings, including PP
nonlinear preconditioning, before adapting the code as our likelihood/score
reference. This requires its own bounded reproduction plan. No new research
runs were launched for this clarification; no rejection of the published
method is supported. The previous commit completed mechanics, not this
publication-validation obligation.
