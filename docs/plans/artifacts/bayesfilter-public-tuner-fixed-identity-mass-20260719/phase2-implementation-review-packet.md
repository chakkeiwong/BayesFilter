READ-ONLY REVIEW PACKET

Objective: review the fixed_identity mass policy and canonical NeuTra route migration for correctness and boundary safety.
Changed paths: bayesfilter/inference/hmc_kernel_tuning.py, bayesfilter/inference/hmc_tuning.py, bayesfilter/inference/neutra_end_to_end.py, bayesfilter/__init__.py, bayesfilter/inference/__init__.py.
Evidence: py_compile passed; focused tests passed 212 passed, 1 skipped; added contract tests passed 56 passed, 1 skipped.
Required behavior: fixed_identity constructs identity covariance in caller coordinates, skips mass replacement, preserves signature through Phase 4-7, emits mass_policy in final private/public handoffs, and rejects mutation. NeuTra must call only public tune_hmc_kernel and use BayesFilter-owned replay. Acceptance target 0.70, band [0.65,0.75].
Forbidden: no scientific validity claim, no private specialized tuner in active route, no reconstruction of mechanics in campaign, no fixed 1,000-draw tuner veto.
Question: identify any material implementation, API, mathematical, or route-governance flaw that should block the serious LGSSM validation. End exactly with VERDICT: AGREE or VERDICT: REVISE.
