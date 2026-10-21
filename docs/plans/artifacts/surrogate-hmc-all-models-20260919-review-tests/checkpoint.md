# Active checkpoint — 2026-10-06

Question: verify the optional marginal-mixture LEDH weight repair, ultimately for surrogate HMC.
Branch: ledh-graceful-failure-execution-20260917; baseline f5e69d716.
Current stage: expanded master implementation and pre-run audit; no new campaign launched yet.
Authoritative plan: docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md.
Main chapter now includes chapters/ch19c_marginal_extension.tex; PDF build/inspection pending.
New runner: scripts/run_surrogate_hmc_marginal_expanded.py; support bayesfilter/testing/marginal_expanded.py.
Initial bounded budget: 24 GPU-work hours on idle RTX4080 SUPER GPU UUID ending e805b; not a forecast.
Grid: LG d=1,3,10,20 at T=10,20,50,120,240; existing KSC d=1, range-bearing d=4 at all horizons.
Pending user clarification: meanings of nonlinear d=1,3,10,20 extensions; must not claim them covered.
Completed old evidence: docs/plans/surrogate-hmc-marginal-weight-results-2026-10-05.md.
Next: focused driver/statistical tests and GPU call-chain/reference checks, document build; fix material findings before full launch.
Preserve unrelated staged inference changes. No default-policy or HMC promotion.
