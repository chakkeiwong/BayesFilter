# Structural UKF NeuTra Campaign Close

Date: 2026-07-17

Status: `CLOSED_QUALIFIED_PASS`

The campaign completed target validation, target-specific learned-NeuTra
training, fixed-kernel tuning repair, adaptive warm-up, 4,000 retained draws
per chain, and physical truth-tail evaluation. The terminal classification is
`QUALIFIED_NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS_OWNER_ADJUDICATED`.

The scientific result is limited but positive: learned NeuTra supported
health-valid transformed HMC on the Chapter 18b structural-UKF target, modern
R-hat passed with folded diagnostics included, tail ESS was strong, the owner
accepted bulk ESS `971.06` as sufficient relative to the arbitrary original
`1000` convenience threshold, and all five noncentral generating parameters
had `p_truth >= 0.28442`.

Execution drift review:

- no drift in target, data, transport payload, GPU/XLA route, model, truth-tail
  rule, R-hat rule, or tail-ESS rule;
- tuning repairs were recorded in fresh attempt roots and preserved failed
  evidence;
- the original bulk-ESS threshold was changed only by explicit owner direction
  after observing the result and is disclosed as post-result adjudication;
- the final GPU process was stopped after preserving the complete 4,000-draw
  retained archive when cumulative campaign compute exceeded the planned
  eight-hour cap;
- no uncompleted third retained chunk is used;
- no second data seed is required under the truth-tail rule;
- LaTeX documentation states both the positive result and its limitations.

No further structural run is required for the present limited claim. Broader
calibration, reliability, exact-likelihood, superiority, or default-readiness
claims remain outside this campaign.

