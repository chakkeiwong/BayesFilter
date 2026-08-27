# Phase 8 Raw Measure and Ledger Audit Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_MEASURE_AUDIT_METADATA_BOUND`  
Budget cap: `1800 s` within the unchanged global `64800 s` cap  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase8`

## Objective

Audit the stored q=20 M0 finite computation independently of its summary
markdown. The audit checks that the hash-bound protocol, raw tensor receipts,
stage sequence, cumulative log-mass ledger, terminal weights, mutation
acceptance counts, and signed mode statistic describe one consistent measure.
It is a bookkeeping and measure-identity gate, not a proof of SMC-U unbiasedness
or mode discovery.

## Entry gate

Phase 6 must first pass the corrected acceptance receipt repair and produce at
least one metadata-bound pilot with `mode_axis=2` in the protocol. The exact
fixture must remain passing. Legacy receipts without that field are retained as
historical candidate evidence but cannot pass this audit.

## Skeptical audit

| Risk | Misleading success mode | Check | Interpretation |
|---|---|---|---|
| Caller-stamped hash | metadata says one protocol while tensors came from another | recompute canonical hash and every file digest | hard gate |
| Broken stage ledger | final mass is finite but resampling increments are missing or reordered | stage indices, beta monotonicity, cumulative residual | hard gate for ledger consistency |
| Wrong terminal weights | weights are normalized but do not match the stored final increment | recompute terminal softmax from target/proposal logs | hard gate |
| Acceptance denominator | rate looks plausible but counts coordinates instead of particles | accepted/proposal count identity | hard gate for receipt validity |
| Mode-axis drift | downstream split uses a different coordinate from the pilot | explicit protocol field and raw-tensor recomputation | hard gate for mode diagnostic, not a discovery theorem |
| Finite measure mistaken for target law | all identities pass on a finite cloud | explicit nonclaims and separate scientific ledger | cannot promote authority |

## Evidence contract

| Field | Predeclared choice |
|---|---|
| Question | Does the stored M0 artifact faithfully encode the finite proposal/target computation it claims? |
| Comparator | Metadata-bound M0 pilot receipt; legacy receipt is a negative control and must be rejected for missing mode metadata |
| Primary criteria | all audit gates pass: protocol/hash, receipt digests, shapes/finite values, stage order/beta, mass ledger, terminal weights, acceptance counts, mode-axis statistic |
| Vetoes | any mismatch, missing metadata, non-finite tensor, missing stage, or overwritten output |
| Explanatory diagnostics | mass value, ESS, mode fraction, acceptance, root counts; none establish target correctness |
| Nonclaims | no unbiasedness theorem, exhaustive mode discovery, posterior correctness, IID whitening, HMC readiness, or default promotion |
| Artifact | audit JSON/Markdown, input hashes, command/environment manifest, decision and inference tables |

## Execution

Run the CPU-hidden auditor
`docs/benchmarks/run_ssl_lstm_q20_particle_authority_measure_audit_2026_08_25.py`
on each metadata-bound Phase 6 root. Run once on a legacy root as a declared
negative control and require the expected metadata failure. Do not modify input
receipts. Preserve unique output roots.

## Refresh and continuation

If all audit gates pass, refresh the next subplan toward a reviewed NeuTra
rerun using the corrected mode axis and the metadata-bound bank. If a gate
fails, repair the smallest ledger/metadata cause and rerun the same input. A
finite audit failure is a repair trigger; stop the whole program only under the
master real-blocker definition.
