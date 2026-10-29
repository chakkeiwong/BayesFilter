# Automatic continuation of existing phase 3

The serial continuation scripts/run_surrogate_hmc_consumer_followups.py waits
for the current advance.json to be terminal and for its campaign ledger to have
no running reservation. It launches only paired consumer diagnostics from
existing fresh selections, preserving each model's recorded contraction,
precision and partition offset. It neither retunes on claim data nor selects
another target. Current source hashes remain checked by the normal consumer
selection loader. Missing selections are recorded for repair, not replaced.

The scientific contract remains consumer-paired-contract.md. Each worker
uses the existing master run command, which enforces the shared worker-wall and
attempt budgets, with a 2400-second cap. New model outputs never overwrite
earlier evidence. The controller itself waits without initializing TensorFlow;
its trusted process takes GPU/process observations around each comparison.
Empty NVIDIA process observations do not establish GPU exclusivity.

Skeptical audit: baseline is the preserved pre-repair complete batch source;
inputs and controls match the fresh candidate scope. Materiality, finite
force and validity are pass/fail criteria; timing is descriptive. A missing
selection or rejected candidate does not cancel independent models. No
posterior, canonical, oracle, or statistical superiority claim is issued.
The controller updates next-phase-plan.json after every phase.

Validation: four CPU orchestration checks cover terminal-parent waiting,
active-reservation exclusion, exact fresh-scope command wiring, continuing
after a failed model and exhausted-budget no-launch. Together with the
existing master suite, 148 tests pass in automatic-followup-tests-01.log.
CPU devices were intentionally hidden with CUDA_VISIBLE_DEVICES=-1.

Command: absolute tftwogpu Python, scripts/run_surrogate_hmc_consumer_followups.py,
--campaign [this directory] --output [this directory]/consumer-paired-followups-01
--wait-seconds 36000 --per-model-seconds 2400.
