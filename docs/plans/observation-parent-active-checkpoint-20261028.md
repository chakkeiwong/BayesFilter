# Active checkpoint
Question: document and default conditional/observation-weighted parents; compare
SIR likelihood and analytical scores T10/20/50 against saved Zhao-Cui fits.
Checkout ledh-iapf-execution-20260922; branch ledh-refactor-with-policy-fix;
base b543565bed052016086059c929d0648e205fb4a0. Preserve unrelated changes.
Plan: observation-parent-default-and-sir-accuracy-20261028.md.
Result: observation-parent-sir-accuracy-result-20261028.md.
Evidence: artifacts/observation-parent-sir-accuracy-20261028/analysis01/summary.json.
Done: default implementation, scoped compatibility, analytical/FD/parity tests;
all 1584/1584 rows, zero invalid filters; final six source hashes match launch.
References usable20/33 cells: all +/-10% rates and noise1.25T10 fail overlap.
CW observed score RMSE lower than original in20/20 usable cells per component;
no CW/original MSE ranking meets the predeclared paired interval+sign criterion.
CW versus conditional-uniform is mixed: noise0.8T20 noise score favors CW;
noise1.25T50 noise score favors uniform (exploratory paired evidence).
Simple controls sometimes outperform CW, so accuracy/default-readiness promotion
is vetoed. Owner-selected proposal default is a weight-concentration decision.
Reference is approximate; fixed-data FP64 diagnostic, locally fixed categorical
parents, not canonical admission or HMC readiness. No independent Claude review
(login unavailable); skeptical local review recorded.
Successful numerical runtime6342.28s; tests/startup/review charge361.27s;
conservative reporting allowance300s; total charge7003.55s of7200s.
No further experiments. Manuscript Sections16.8–16.11, full result note,
paired analysis, figures, PDF and final provenance are complete.
Delivery: artifacts/observation-parent-sir-accuracy-20261028/delivery-final/manifest.json.
ESS follow-up: weighted time-mean and time-minimum ESS higher in160/160 paired
runs for20 usable cells and264/264 for all33. Section16.11 states these counts;
delivery-ess-audit01 preserves the audit and updated document separately.
Next research question, requiring a new bounded plan: fresh parameter-specific
reference fits for wide rate shifts and more paired streams in usable cells;
then scope-specific tuning and independent-data/production-precision evaluation.
