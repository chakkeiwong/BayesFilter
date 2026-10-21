# Transport arithmetic probe result

Fixed exponential caching and combined logsumexp/softmax both passed primitive
arithmetic checks but failed the predeclared cost nomination. Warm candidate /
baseline ratios were 1.093 and 1.197 for caching (KSC, LGSSM), and 1.078 and
1.024 for combined reductions. Five CPU value, all-direction derivative and
underflow/nonfinite tests pass. Individual GPU repeats and source snapshots are
preserved in the four probe directories and transport-primitive-review.json.
These are descriptive primitive results; they do not establish a statistical
speed ranking or whole-filter regression. No working scientific source changed.
The cache construction cost was not measured: the cache already failed when
that nonnegative extra cost was omitted, so it cannot rescue this nomination.
Memory comparison was not established; aggregate allocator peaks are recorded.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject both candidates for this cost repair | Arithmetic passes, useful speed nomination fails | No numerical hard veto | Microbenchmark fusion and timing variability | Inspect tile scheduling in existing phase2 | No scientific rejection of transport/HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | Arithmetic passes at tested synthetic inputs |
| Statistically supported ranking | None |
| Descriptive-only differences | Warm latency ratios above |
| Default readiness | Neither candidate retained |
| Next evidence | A distinct schedule candidate needs its own paired whole-filter checks |

Red team: the microbenchmark may miss a whole-filter benefit, but these probes
provide no positive nomination for spending a full five-model retuning cycle.
Their failure is not evidence that all transport implementations are slow.
