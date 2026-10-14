# Phase19 result: resampling amplifies rounding; d2 compilation also changes

Completed2026-09-23. Four GPU workers perform228 selected replays. The source
and152 source/50 output hash checks pass. FP64 and ordinary FP32 reproduce
all57 original selected likelihoods. TF32 reproduces36/57; all21 selected d2
cases differ. One predeclared control using the original warm-up sequence does
not restore the21 old values. The maximum initial replay log-likelihood
discrepancy is0.40256. Neither failed replay is silently accepted as the old
execution. All traces are bitwise neutral and every recorded ancestor/mixture
choice agrees with independent host reconstruction.

The numerical source is unchanged. Matched optimized-HLO operations show10
backend configuration differences in the d2 fitted kernel, including cuBLAS/
Triton switches. The two fresh TF32 executions themselves differ on17/57
values; five pairs change a discrete decision. Their maximum cloud difference
before a changed decision is0.00073195, rising to0.98718 afterward. Warm-up
order alone is therefore not a sufficient reproducibility remedy. The backend
choice is observed; its precise selection mechanism is not established.

## What the valid replays explain

There are72 TF32/reference pairs with exact original values. Sixty-seven
change a discrete decision:65 first change an ancestor and two a Gaussian/
floor mixture choice. Maximum cloud error before the first choice change is
0.00161803; afterward it reaches1.80968. These are selected difficult seeds,
not a frequency estimate or a statistical ranking.

The dominant first long-horizon stress seed,2030100483, exactly reproduces
phase18. A CDF difference of only9.7531e-5 at the first observation changes one
of128 selected ancestors. From the actual rounded CDFs, the exact23-bit-lattice
probability of an individual ancestor disagreement is0.00310934 (expected
0.398 changed ancestors for128 draws). The maximum cloud difference before
that change is0.00102703; afterward it reaches1.51212. Both executions later
have minimum particle ESS about1.0605. Their normalized likelihood estimates
are558.2207 in TF32 and32.8433 in FP64; this one pair supplies99.39% of the
original contrast's centered squared variation.

This directly locates the amplification: a small numerical threshold change
copies a different particle, and subsequent concentrated weights amplify that
changed ancestry. It does not prove systematic TF32 bias or an invalid iAPF
identity. The poorly adapted random guide is an explicit stress construction.
All phase18 mean-precision and strict-path vetoes remain unchanged.

For a lattice of size L=2^23, label j occupies integer interval
[ceil(L F[j-1]),ceil(L F[j])). The agreement probability under shared draws
is the total length of equally labelled interval intersections divided by L.
This calculation uses both actual CDFs and the consumer's right-sided search
and final-index clamp. It proves the conditional threshold sensitivity; it
does not supply a population likelihood-error bound.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
|---|---|---|---|---|---|
| Explain36 exact TF32 replays | Original values and neutral traces verified | No replay veto for these cases | Outcome-selected cases | Preserve explicit ancestry mechanism | Population failure frequency or bias |
| Keep21 old d2 replays unresolved | Exact old-value reproduction fails twice | Causal attribution to those old paths vetoed | Original compiled executables no longer available | Fresh disjoint d2 comparison with every contemporaneous trace | Warm-up fixes reproducibility |
| Preserve phase18 decision | No new precision sample here |37 mean-precision and11 old strict checks remain unresolved/failed | Tail uncertainty and compiler context | Phase20 recorded-path comparison | Broad TF32 or default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Old d2 value replay fails; actual trace neutrality and choice reconstruction pass |
| Statistically supported ranking | None; outcomes were selected for influence |
| Descriptive-only differences | All seed-specific errors, ESS and changed-path counts |
| Default-readiness | Not established; no numerical default changed |
| Next evidence needed | Fresh streams and complete traces from the actual compiled processes, with unchanged precision criteria |

Execution: `diagnose_iapf_tf32_tail_replay.py --execute`, followed by the one
allowed `--execute --warmup-original` control, both through the escalated
tftwogpu Python. Each attempt manifest records Git89065bc6, source closure,
exact command, seeds, GPU UUID, TensorFlow/build, XLA, growth, allocator and
wall time. Selection binds the original fixtures and CSV hashes. Initial
results are preserved in `initial-compilation/`; actual second-compilation
paths are in attempt04. `compiler-difference.json`, `localization.json`,
`cross-compilation/` and `verification.json` support the interpretations above.

Phase GPU charge128.674175 seconds. The settled `budget.json` includes CPU
preflight/review charges and leaves approximately18.029 CPU and47.189 GPU
hours, no reservations. Phase20 uses its own fresh ledger and output root.

Post-run skeptical review: a different legal floating-point backend can change
rounding without implementing a wrong real-arithmetic formula. The current
evidence supports compiler-sensitive numerical execution, not a new proven
compiler wrong-result defect. Large differences before any changed sampling
decision, or failure of independent arithmetic controls, would require a
different remedy. The weakest evidence remains reconstruction of the21 old
d2 paths; preserve that gap and collect full traces during the fresh run.
