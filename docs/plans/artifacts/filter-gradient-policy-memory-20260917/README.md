# Audit evidence

Authoritative results for this audit:

- `static-complete.json.gz`: all tracked Python discovery, AST facts, NumPy
  calls, defaults, imports/lazy exports, and typed call/callback-reference edges.
- `static-complete.md`: owned-module index and neutral syntax counts.
- `reviewed-call-chains-complete.json`: seven resolved source chains supporting
  the findings. These are static wiring evidence, not executed endpoint proof.
- `memory-v3/*-{off,on}.json`: eight fresh GPU worker manifests and outputs.
- `memory-v3/*.optimized_hlo.txt`: optimized HLO for the four XLA arms.
- `memory-comparison.json`: elementwise parity, graph-boundary checks and memory.
- `audit-manifest.json`: final tools, commands, source identity and artifact hashes.

The result note is `../../filter_gradient_policy_memory_result_20260917.md`;
the algorithm inventory is `../../filter_gradient_algorithm_inventory_20260917.md`.

The following are **superseded preliminary artifacts**, retained for recovery:

- `static-audit.*`: old limited scan with overbroad keyword/loop classification;
  its candidate count is not a violation count.
- `static-v2.*`, `static-v3.*`, `static-final.*`: intermediate inventory versions,
  before all ownership/reference-edge refinements. Large plain JSON snapshots
  were losslessly gzip-compressed and verified against their original bytes.
- `reviewed-call-chains.json`: initial direct-call-only traversal with incomplete
  callback handling and an incorrectly named APF endpoint; use the complete version.
- `memory/gpu-jit-off-v2.json`: tiny off-only preliminary run with a misplaced
  trace-stage snapshot; excluded from the paired comparison.

The initial failed GPU initialization is described in the result note; no
result JSON was emitted by that failed worker. The first corrected rectangular
off worker's structured JSON was retained; its console output was returned by
the execution tool rather than saved to a separate log. Other corrected
workers have both JSON and console logs. Do not infer missing measurements
from an absent preliminary log, or use old heuristic counts as policy verdicts.
