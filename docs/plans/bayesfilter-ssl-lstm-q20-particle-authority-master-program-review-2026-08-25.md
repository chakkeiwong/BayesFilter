# Skeptical Review of the SSL-LSTM q=20 Master Program

Date: 2026-08-25  
Reviewer role: skeptical developer / scientific audit  
Artifact reviewed:
`docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`

## Verdict

`PASS_FOR_PHASE_0_EXECUTION`

The program is executable only through its explicit gates. It does not certify
the particle authority, any modular arm, NeuTra, or HMC in advance.

## Findings first

| Risk checked | Finding | Repair/status |
|---|---|---|
| Wrong baseline | Historical normalized replay is explicitly descriptive; C0 and M0 are separate | Pass |
| Proxy promotion | Whitening, ESS, loss, covariance, and runtime are explanatory only | Pass |
| Missing stop conditions | Continuation vetoes and a concrete real-blocker definition are stated | Pass |
| Unfair comparison | Target, partitions, seeds, protocol hash, tuning scope, dtype, and budget are bound | Pass; verify in runners |
| Stale context | Master reconciles the two review files and repairs the old plan status | Pass after Phase 0 patch |
| Hidden defaults | Numeric choices carry provenance and remain hypotheses/warm starts | Pass |
| Environment mismatch | CPU fixture and GPU NeuTra lanes specify env/device/memory/XLA records | Pass; runtime check required |
| Artifact mismatch | Every phase has unique roots, manifests, result, repair, and refresh outputs | Pass; fail closed in runner |
| Phase drift | Inter-phase protocol requires classification, repair, and next-subplan refresh | Pass |
| Budget overrun | Phase caps sum to the 18-hour cap; reserve cannot silently cover a failed gate | Pass |
| Mathematical overclaim | ETPF/GenUT/ET-PF roles are finite-moment/quadrature/approximate; M0/M3 contracts are conditional obligations | Pass |
| NeuTra gate bypass | NeuTra is downstream, batch-native GPU/XLA, and HMC is excluded | Pass |

## Material assumptions still unproved

The program correctly leaves these as execution obligations: M0 conditional
unnormalized-mass identity, q=20 mutation invariance, finite defensive-tail
second moment, complete replay metadata, LEDH invertibility, and finite-run mode
reachability. Their absence is not hidden; Phase 1 and Phase 2 are designed to
test them.

## Decision table

| Decision | Primary criterion | Veto | Uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Start Phase 0 | documentary and environment closure | stale paths or import failure | none yet | run focused checks and write result | no scientific validity |
| Start Phase 1 | Phase 0 gate passes | missing runner/schema | fixture tolerance is measured, not assumed | implement/run known-density fixtures | no q=20 authority |
| Start Phase 2 | M0 fixture and support gates pass | exact contract failure | particle/mutation feasibility unknown | fresh paired C0/M0 pilot | no mode-discovery theorem |
| Start Phase 3 | viable M0 candidate | invalid authority metadata | arm ranking underpowered | one-factor arms | no superiority |
| Start Phase 4 | valid input bank | GPU/batch/parity failure | training outcomes unknown | scoped GPU screen | no HMC/posterior claim |

## Conclusion

No material plan flaw requires user direction before Phase 0. Candidate failure
must trigger the documented repair loop. Stop only on a stated real blocker or
budget exhaustion.

## MathDevMCP audit record

The local MathDevMCP CLI was run before Phase 0 execution:

```text
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/.venvs/mathdevmcp-mcp/bin/python -m mathdevmcp.cli \
  audit-and-propose-assumptions \
  "What assumptions are required before a fresh q=20 SMC/SMC-U route can be treated as an auditable particle authority?" \
  --target "E[gamma_hat_t(f) | frozen protocol] = integral tilde_pi_t(theta) f(theta) dtheta" \
  --output docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-mathdevmcp-assumption-audit-2026-08-25.md
```

MathDevMCP returned `proposal_ready` but marked the direct target as
`human_review_required`: its bounded rule set could not infer typed route
assumptions. This is an audit limitation, not evidence that the assumptions are
unnecessary. The program already supplies the required explicit obligations
(frozen protocol, invariant mutation, common support, finite second moment, and
metadata retention) and requires an exact fixture before promotion. The
generated report is preserved at the output path above and is not treated as a
proof certificate.

Two additional bounded checks classified the proposal claim as an ambiguous
report/status boundary and the finite-cloud equality as requiring semantic
human review. Those results reinforce the program's nonclaim boundary: source
identities and finite-moment equations must be checked with typed fixtures and
source anchors, not promoted from a symbolic-router status.
