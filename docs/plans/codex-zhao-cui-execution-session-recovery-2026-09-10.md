# Zhao-Cui Execution Session Recovery, 2026-09-10

Status: investigation complete; 28 focused CPU tests pass. Scientific audit,
guide validation, and Algorithm 3 GPU/XLA execution remain unfinished.
Scope: recover the execution session identified in the owner's attachment,
explain the stall, and establish the saved implementation state. This is an
incident investigation, not completion or scientific approval of Algorithm 3.
All times below are Hong Kong time (UTC+8), September 10, 2026.

## Correct Session

- Thread: `01a08af9-04e6-7ab3-bedf-7f52c2821b4d`.
- Saved rollout: `/home/chakwong/.codex/sessions/2026/09/10/rollout-2026-09-10T18-59-30-01a08af9-04e6-7ab3-bedf-7f52c2821b4d.jsonl`.
- Codex: `0.153.4`, `gpt-6-astra`, reasoning effort `max`, legacy history.
- Session cwd: `/home/chakwong/BayesFilter`.
- Research checkout: `/home/chakwong/BayesFilterZhaoCui`.
- Recorded research base: `47176bce7cdaa91ddd4466c39f90a11ad005c800`.
- Temporary stage: `/tmp/zhao-cui-algorithm3-audit-20260910/`.

This is the session that received `do as you suggested` at 19:28 and performed
the implementation work. It is distinct from the earlier failed audit thread
`01a086f1-70ca-7663-8775-a17029b3e869`, documented in
[the previous recovery note](codex-zhao-cui-audit-session-recovery-2026-09-10.md).
The saved execution session also contains later activity absent from the
attachment: a request to change execution order at 22:46 and a manual
compaction attempt at 23:02.

## Failure Evidence

| Time | Event | Interpretation |
| --- | --- | --- |
| 19:43:26 to 21:08:05 | One tool batch launches CPU tests and requests escalation for a TensorFlow GPU probe. The corresponding approval arrives at 21:08:05. | The long tool gap includes an approval wait. The CPU test itself reports 9.48 seconds; this was not an 85-minute numerical run. |
| 21:13:23 | Tool batch finishes; context counter is 228181. | Below the 244800 compaction threshold. |
| 21:14:38, 21:15:46 | Ordinary sampling requests fail with `Upstream request failed`; a subsequent request succeeds. | Model response streaming was already failing before compaction. |
| 21:16:34 | Final source fixes are copied successfully; context counter is 231119. | Still below the compaction threshold. |
| 21:17:48 to 21:23:50 | Sampling retries reach 5/5; the turn ends without a final answer. | Initial execution stop is response-stream failure, not automatic compaction. |
| 22:47:18 | A master-plan patch fails verification. Context reaches 249765 against threshold 244800. | The retry now needs mid-turn compaction. Full context limit 258400 is still not reached. |
| 22:48:26 to 22:50:49 | Automatic compaction retries fail with `Upstream request failed`; turn ends. | Compaction prevents further work. |
| 23:02:36 to 23:05:09 | Manual compaction also fails twice; its third request is interrupted. | No successful compaction is saved. |

Runtime requests went to `https://llm.visioncoder.ai/responses`. Successful
HTTP 200 headers preceded failed streams; those headers do not mean the model
response completed. One manual-compaction error supplied provider request ID
`202609101502389500409018268d9d67AWOr7U5` for support investigation.

The evidence locates the failure in the model gateway/upstream response path.
It does not identify the server-side defect or establish that context size
caused the ordinary stream failures. Changing numerical execution order can
improve checkpointing but cannot guarantee this service will complete a
request. Repeated recovery/file-reading work in the same conversation grew
context from 44027 at 19:00 to 120334 before implementation began and 249765
on the final continuation.

All 298 saved JSONL records parse. All 42 top-level tool calls have matching
outputs, with no orphan outputs. A matched output can still report a failed
operation: the final master patch is one such failure. No successful
`compacted` record exists. The final event is `turn_aborted`, not a running
scientific job. An escalated host process inspection at 23:24 found no
Zhao-Cui/C2 experiment, pytest, Lean, or LaTeX process.

## Preserved Work

Every one of the ten files in the temporary stage is byte-identical to its
counterpart in the research checkout. The saved work is therefore not confined
to `/tmp`. Relevant files, relative to the research checkout, are:

- `bayesfilter/highdim/zhao_cui_algorithm3_tf.py`
- `bayesfilter/highdim/zhao_cui_algorithm2_preparation_tf.py`
- `bayesfilter/highdim/gaussian_hermite_tt_transport_tf.py`
- `bayesfilter/highdim/hermite_gram_tf.py`
- `bayesfilter/highdim/transport.py`
- `bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py`
- `tests/highdim/test_zhao_cui_algorithm3_tf.py`

The execution note also survives as
`docs/plans/bayesfilter-zhao-cui-algorithm3-audit-repair-2026-09-10.md` in the
research checkout. It still describes the entry plan rather than a completed
validation result.

The master program and LaTeX source match the baseline hashes saved before
this implementation attempt:

| File | SHA-256 |
| --- | --- |
| `docs/plans/bayesfilter-zhao-cui-algorithm3-ukf-guided-tt-master-program-2026-09-09.md` | `7ad7d08a615fedd1e3a05924716ce95e65780a4b2c74f8701b6f4dcbf9a5e280` |
| `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex` | `8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69` |

The attempted master edit at rollout line 291 would clarify the frozen score
target, classify ESS as explanatory, and correct the description of retained
versus full adjacent TT capture. Line 293 records `apply_patch verification
failed` because an expected context line was absent. None of that patch was
applied; the master still needs reconciliation with the new code.

## Test Evidence And Bounded Verification

Saved test results are stage-specific:

- Initial seven-test suite: 5 passed, 2 failed in 9.48 seconds.
- After first repairs, Algorithm 3 plus C2 proposal tests: 13 passed in
  8.31 seconds.
- Expanded Algorithm 3 plus transport regressions: 20 passed, 2 failed in
  10.71 seconds. Failures were an incorrect Hermite-basis import in a test and
  a changed manifest field expected by an existing transport regression.
- Both latter fixes were copied at 21:16, together with an offline
  likelihood-weighted sigma-point guide. No subsequent pytest result exists
  in that session. The guide has no direct dedicated test in the saved suite.

The saved RTX 4080 SUPER probe reports TensorFlow 2.19.1, verified memory
growth, and a matrix-product sum of 54. This is device-access evidence only;
it is not an Algorithm 3 GPU/XLA smoke or a C2 pilot.

Skeptical preflight for the recovery check: rerun the existing focused tests
against the actual checkout because the final fixes were not verified. The
comparator is their existing assertions, not a scientific baseline. Pass/fail
means regression status only. These are small CPU/reference checks, with GPU
devices deliberately hidden and non-XLA execution explicitly selected by the
Algorithm 3 tests. No runtime default changes or scientific ranking follow.
Allow one invocation, at most three minutes; preserve its output and exit
status here. No source edits, GPU campaign, or environment changes are needed.

Command executed, cwd `/home/chakwong/BayesFilterZhaoCui`:

```sh
CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_TEST_DEVICE_SCOPE=cpu BAYESFILTER_PRELOAD_CUSTOM_OP=0 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 MPLCONFIGDIR=/tmp/zhao-cui-recovery-mpl-20260910 timeout 180 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -p no:cacheprovider -q tests/highdim/test_zhao_cui_algorithm3_tf.py tests/highdim/test_c2_gaussian_hermite_proposal_tf.py tests/highdim/test_transport.py tests/highdim/test_p83_minimal_source_route_transport_slice.py
```

Recovery verification completed with exit code 0:

```text
............................                                             [100%]
28 passed, 2 warnings in 11.85s
```

Both warnings are TensorFlow Probability's existing `distutils` deprecation
warnings. No repair was needed in this investigation. This rerun verifies the
saved final fixes under these CPU/reference tests. It does not validate the
new guide directly, GPU/XLA behavior, C2 filtering quality, or the full
scientific program. The guide is not exercised by these tests, and both
planned Algorithm 3 artifact directories (`zhao_cui_algorithm3_20260909` and
`zhao_cui_algorithm3_20260910`) are absent. The manuscript PDF and Lean source
also remain present; their compilation/proof verification was not rerun.

## Continuation Checkpoint

Use this note and the actual research checkout to continue; importing the
entire failed conversation recreates its context burden. Neither session
database edits nor another manual compaction are prerequisites for proceeding
from the saved files. The original session and configuration remain intact.

1. Reconcile the master with the execution note and actual code. Its attempted
   edit failed. State that the initial analytical derivative holds fitted
   coefficients, charts, uniforms, sampled paths, and proposal densities
   fixed. Its target is the finite importance likelihood, not the adaptively
   refitted TT likelihood or exact model likelihood. Resolve the contradictory
   ESS roles and retain the paper/author-code checks required by the master.
2. Complete focused validation of `likelihood_weighted_sigma_point_chart`
   and its connection to recursive preparation. The existing 28 tests pass,
   but they do not call this newly added guide. Save each completed check and
   its result before moving to another stage.
3. Continue the previously authorized bounded GPU/XLA smoke and C2 diagnostics
   only after their implementation and evidence prerequisites hold: first
   smoke at most 15 minutes, C2 fits at most 30 minutes total, at most four
   recorded GPU/C2 launches, fresh output directories, and no T=20 ladder.
   The recovered runs are CPU tests and device probes, not completed campaign
   launches. Recheck host/device occupancy before launching.

Keep file reads and returned logs bounded. Write command output to a versioned
local result and return a short summary. Keep approval-bearing device probes
separate from unrelated test collection so one pending request does not delay
collection of an already-finished test. A completed stage needs an on-disk
result, not just an intention to checkpoint. These changes reduce lost work
and repeated context growth; they do not repair the upstream streaming service.

## Evidence Anchors

Saved rollout lines: 217/219 (long tool batch), 229 (first tests and GPU
probe), 250 (13 passes), 257 (RTX 4080 SUPER probe), 264 (20 passes and two
failures), 269/272 (final source fixes), 281 (failed execution turn ends),
286 (execution-order request), 291/293 (failed master patch), 295 (failed
automatic-compaction turn ends), 298 (manual-compaction interruption).

Read-only runtime log database: `/home/chakwong/.codex/logs_2.sqlite`.
Relevant `logs.id` values: 456875744 (GPU command approval), 456877784
(228181 tokens), 456878816/456879131 (early sampling retries), 456879331
(231119 tokens), 456879973 through 456881189 (five subsequent sampling
retries), 456896452 (249765 tokens), 456896631/456896788 (automatic
compaction failures), 456898789/456898960 (manual-compaction failures),
456899069/456899070 (interruption).

Official documentation retrieval was unavailable: web search/open returned
503, and direct retrieval returned 403 both inside and outside the sandbox.
The diagnosis relies on local session and runtime evidence. No Codex
configuration, rollout, session database, or research source has been modified
by this investigation.
