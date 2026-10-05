# Mixed KR/TTSIRT transport execution result

The active `FixedTTSIRTTransport` public numerical methods now default to
complete XLA execution, including the previously eager logarithms in
`potential`, `proposal_log_density` and `log_normalizer`. Potential and proposal
log density share one retained native program that returns both signs of the
same logarithm. Their wrappers only select a dictionary field. No PDF
expression, grid, bisection setting, status code, random stream or numerical
tolerance changed; legitimate `log(0)` infinities remain valid API results.

Static package discovery finds no constructor/caller of the old scalar
`KRTransport`; its docstring now explicitly marks it historical diagnostic-only.
Private bisection/reference-density helpers on `FixedTTSIRTTransport` also
retain loops but are not reached by the inspected native consumers. This is
bounded call-chain coverage, not a declaration that the entire file has no
Python loops or that external/dynamic callers cannot exist.

| Gate | Runs | Result |
| --- | --- | --- |
| Historical classification and public method discovery | 04070, 04072; final 04092 CPU / 04093 GPU | Pass; unclassified new public methods fail discovery |
| Frozen original values, query/core pullbacks and directional finite differences | 04092 / 04093 | Pass at unchanged 1e-10 bound |
| Default XLA, enclosing graph, changed operands, one trace, exact replay and HLO | 04092 / 04093 | Pass; six final checks per backend |
| Zero density, nonfinite inputs and normalizer status | 04092 / 04093 | Pass |
| Active native TTSIRT regression suite after first wrapper repair | 04073 / 04074 | Pass; precedes final shared owner |
| Source-runtime consumers after final shared owner | 04106 / 04107 | 12 checks per backend pass |
| Source-policy/campaign checks | 04108 | 129 pass; 270 sources / 1,424 exact allowances |
| Fresh-process values, input identity, source hashes and costs | 04094--04105 | Pass; all 3,391 source hashes identical |

Frozen original transport/density/TT sources at `3582b4ac` are the independent
value/pullback comparator. The cost baseline instead uses public wrappers at
`5bbce48b5` with the current native density dependency: it is partially compiled,
not an entirely eager density implementation. Maximum complete-output error
in the cost cohort is `2.220446049250313e-16` under the unchanged `1e-10` bound.

All three public log methods are measured together, d=2, deterministic
CPU-generated sample counts 3/128, 20 synchronized replays, fresh processes.

| Device | Samples | Arm | Cold s | Median warm ms | RSS after compile MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU | 3 | previous | 0.575 | 2.186 | 744.0 |
| CPU | 3 | graph | 0.418 | 3.923 | 608.8 |
| CPU | 3 | XLA | 0.628 | 2.492 | 744.5 |
| CPU | 128 | previous | 0.618 | 4.441 | 748.9 |
| CPU | 128 | graph | 0.427 | 6.400 | 609.0 |
| CPU | 128 | XLA | 0.630 | 4.633 | 749.3 |
| GPU | 3 | previous | 1.011 | 3.467 | 1070.4 |
| GPU | 3 | graph | 1.678 | 12.318 | 1068.8 |
| GPU | 3 | XLA | 0.991 | 3.669 | 1069.4 |
| GPU | 128 | previous | 1.079 | 5.345 | 1070.5 |
| GPU | 128 | graph | 1.726 | 15.298 | 1069.2 |
| GPU | 128 | XLA | 1.045 | 5.362 | 1069.2 |

The first preserved cohort 04079--04090 compiled separate potential and
log-density owners and showed about 22 MiB CPU / 31 MiB GPU extra RSS over its
previous arm. Sharing the owner removes that observed overhead in the
refreshed cohort: XLA is within 0.5 MiB CPU and 1.3 MiB GPU of previous RSS.
CPU XLA still uses about 136--140 MiB more RSS than graph. Replays add at most
76 KiB RSS; they do not test repeated constructors or executable eviction.
Warm timing differences are descriptive and do not establish a speed ranking.

GPU allocator peaks for previous/graph/XLA are 25,856/29,952/26,112 bytes at
3 samples and 49,152/53,760/50,176 bytes at 128. All six arms use GPU UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`; sampled monitoring found no foreign
compute process or monitor error. Brief sharing can be missed. Memory growth
was verified before device initialization. Allocator counters exclude host
compiler and CUDA context memory. No target-scale memory claim follows.

Failures are preserved: 04069 was a diagnostic-marker/AST-indentation harness
error; 04076 compared generated zero-cotangent metadata counters. Both HLO
dumps are retained; normalization changes only the GraphToFunction suffix of
`zeros_like` Const metadata at `dummy_file_name:10`. Adverse tests require
instructions, constants and real source locations to remain exact. The
strongest alternative explanation for favorable cost differences is ordinary
process variation; the conclusion is limited to observed footprint and
functional qualification, not statistically supported performance superiority.

| Decision | Primary criterion | Veto status | Main uncertainty / next action | Not concluded |
| --- | --- | --- | --- | --- |
| Accept this execution repair on repair branch | Values, pullbacks, status and XLA pass | No current fixture veto | Wider public integration and target capacity remain | Whole-program completion or main merge |
| Retain historical KR only as diagnostic | No package callers found | Claim use blocked | Dynamic/external callers outside discovery | Author-scale KR correctness |
| Keep canonical scientific admission separate | No qualifying LEDH rebuild in scope | Unsupported claims remain blocked | Continue registered master repairs | LEDH, Zhao--Cui faithfulness or HMC readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current bounded numerical/execution checks pass; prior failures retained |
| Statistically supported ranking | None |
| Descriptive differences | Shared-owner RSS/cold costs and table above |
| Default-readiness | XLA default verified for these methods; no whole-repository promotion |
| Next evidence | Target-capacity and repeated-constructor tests plus remaining public consumers |

Reproducible summary: `mixed-kr-cost-summary-04105.json`. Raw attempts,
current source snapshot, summary and verifier are preserved in
`sir-mixed-kr-evidence-04108.tar.gz` with receipt
`sir-mixed-kr-verification-04108.json` under the campaign artifact root.
