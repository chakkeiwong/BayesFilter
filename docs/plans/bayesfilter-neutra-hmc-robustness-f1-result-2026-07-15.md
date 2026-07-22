# NeuTra HMC Robustness Phase F1 Result

Date: 2026-07-15  
Decision: `PASS_F1_ENGINEERING_VALID_FROZEN_CANDIDATE`

## Outcome

Three 500-step target-specific screen arms ran on common training and held-out
seeds using the new fixture, a target-specific mass factor, and a center offset
from the truth/prior center by one quarter prior scale with alternating signs.
All arms passed GPU/XLA, finite/status, parity, memory-growth, batch-native, and
no-NumPy/host-callback gates.

The inherited wide recipe had the lowest held-out reverse-KL mean (`61.4933`).
The source-width arm differed by `0.0920` with paired MCSE `0.0206`; the lower-LR
arm differed by `0.3684` with paired MCSE `0.0332`. The predeclared rule
nominated `inherited_wide_lr5e3`. These eight-batch differences are
nomination-only and do not support a general method ranking.

A new seed `(20260715, 8201)` then trained the nominated recipe for 5,000 steps
without reusing screen weights. The final artifact passed:

- one compiled GPU/XLA `tf_while_loop`, batch size 128;
- all 501 recorded target evaluations finite/status-valid with zero floor count;
- exact frozen transport, log-determinant, pullback-score, and logdet-score
  parity;
- TensorFlow memory growth configured before device initialization;
- no repository NumPy, host callback, scalar fallback, row map, or sample-axis
  Python loop; and
- frozen payload hash
  `cab56a88caabe557ff8287f399902beddf839f4d43c482c4c132dc46075a5920`.

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Target-specific training protocol | passed |
| Engineering-valid frozen candidate | passed |
| Hard veto screen | passed; no training/status/parity/runtime veto |
| Statistically supported recipe ranking | none |
| Descriptive-only differences | screen heldout means, MCSEs, losses, gradients, runtime |
| New-fixture NeuTra posterior claim | not established; requires F2 |
| Default readiness | not established scientifically |

The strongest alternative explanation is that the held-out reverse-KL proxy
favored a transport that still produces poor HMC geometry or biased posterior
draws. F2 is the discriminating downstream test. F1 does not establish HMC
convergence, posterior correctness, robustness, superiority, production, or
default readiness.

## Artifacts And Handoff

- Screen selection: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f1/screen/selection.json`
- Final training: `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/f1/final/inherited_wide_lr5e3/attempt-01/result.json`
- Terminal consolidated run manifest:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/phase-a/serious_run_manifest.json`.
  The original result payloads preserve contemporaneous seeds, device, GPU/XLA,
  memory-growth, wall-time, and artifact data. Command strings in the terminal
  manifest are transparently labeled as reconstructed from the frozen CLI, not
  as contemporaneous shell transcripts.

Proceed to F2 using the exact frozen payload above and the admitted F0
comparator. Do not retune training or change the target after seeing HMC results.
