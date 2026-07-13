# BayesFilter SSL-LSTM Completion Phase A2 Implementation Review

Date: 2026-07-13 (Asia/Shanghai)

Review class: `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude.

Reviewed exact path:

- `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py`

Reviewed source SHA-256:

- `0dad54c239de11f105f541527447d167114073ab046c796a813b5c1e867452ed`

## Round 1 Finding

The initial bounded review returned `VERDICT: REVISE` for three issues:

1. missing finite-value admission for materialized innovations and forecast
   outputs;
2. seed provenance that could be misread as cross-backend bitwise regeneration
   evidence; and
3. an ambiguous typed `adapter_signature` field containing the accepted A1
   masked-posterior adapter signature.

## Visible Repair

- All free draws and all three materialized innovation tensors are required to
  be finite before terminal or forecast execution.
- All seven eager or compiled forecast tensors are required to be finite before
  provenance construction and return.
- Materialized tensor hashes are explicitly the replay authority. Philox seed
  values are explicitly generation metadata, not cross-backend bitwise
  regeneration evidence.
- Typed provenance uses `a1_adapter_signature`, populated through
  `SSLLSTMPosteriorTarget.adapter_signature()`.

## Round 2 Result

No material findings. The reviewer confirmed that the three prior findings are
resolved and found no new material covariance, static XLA-program, replay, or
provenance defect visible in the exact production path.

This review does not assess runtime success, posterior correctness, predictive
equivalence, calibration, HMC or NeuTra readiness, product/default readiness,
or scientific validity.

## Terminal Trace Audit Repair

The first post-result terminal audit exposed a verifier false positive: an
unanchored `link(` matcher classified read-only `readlink(...)` as a mutation.
The closure verifier itself had passed every member hash and signature check;
the terminal trace audit did not pass until the parser was repaired.

Three bounded review rounds then hardened the exact verifier path:

1. Round 1 required relative-path, descriptor-only, component-safe root, and
   missing mutation-family repairs.
2. Round 2 required fail-closed parse coverage, child/CWD handling or a narrower
   trace contract, truncation rejection, syscall-time symlink resolution, and
   `fchmodat2` classification.
3. The final repair deliberately narrowed the terminal contract to one explicit
   PID, complete nontruncated `strace -yy -s 65535` records, and exactly the
   allowed write-open pattern. Every successful mutation other than
   `open`/`openat`/`openat2` is forbidden; every admitted write open must expose
   a syscall-time resolved descriptor destination inside a component-safe
   allowed root; malformed, empty, read-only-only, multiprocess,
   unfinished/resumed, truncated, or unannotated traces fail closed.

The focused parser suite passed `18/18`. The final bounded review found no
material bypass under this intentionally narrow one-process terminal-log
contract and returned `VERDICT: AGREE`. This is not a general-purpose
adversarial syscall monitor and does not support claims beyond the reviewed A2
closure verifier.

VERDICT: AGREE
