# Source-isolated modern-HMC diagnostic result

The campaign completed on 27 October 2026 at 02:59:46 Asia/Shanghai. Preflight,
all-candidate pricing and qualification exited successfully. Trusted process
inspection found none of the recorded supervisor or worker PIDs still running.
The engineering question in the
[continuation plan](bayesfilter-modern-hmc-isolated-continuation-2026-10-26.md)
is answered for the declared 48-period MacroFinance target and frozen geometry:
all nine records were retained; six candidates passed fresh tuning verification;
and all six were exported and reloaded with their verified endpoints preserved.
Each reload completed eight finite transitions with no recorded health failures.
This short continuation tests serialization and execution, not convergence.

## Candidate results

| L | Epsilon | Terminal disposition | Fresh verification trials |
|---|---|---|---|
| 3 | 0.4 | Verified, exported, reloaded | 64 |
| 5 | 0.4 | Verified, exported, reloaded | 64 |
| 9 | 0.4 | Verified, exported, reloaded | 64 |
| 13 | 0.4 | Verified, exported, reloaded | 128 |
| 18 | 0.4 | Verified, exported, reloaded | 128 |
| 25 | 0.2 | Verified, exported, reloaded | 32 |
| 3 | 0.8 | Movement veto during measurement | Not reached |
| 5 | 0.8 | Non-finite numerical telemetry during measurement | Not reached |
| 25 | 0.4 | Non-finite numerical telemetry during measurement | Not reached |

There were 1,127 unique completed trials across 45 work items: 26 measurement
observations and 19 verification observations. Cumulative observations are not
added together as independent trial counts. The native audit reconciles
7,996,416 charged gradient-work units with its expected completed-work accounting.
All 1,127 attempted numerical chunks returned; none is unobserved. R-hat remained
reporting-only during tuning. All candidate failures remain in the evidence.

## Terminal review

The closeout matched all 1,066 execution-manifest source hashes and confirmed
that all 80 recorded local imports resolve inside the frozen source directories.
The six verified IDs agree across the terminal result, native audit, exported
member index and reload result. Each member identity and verified endpoint match
the corresponding reload archive. GPU memory growth was recorded before device
initialization. The worker used TensorFlow 2.20.0 / TFP 0.25.0, float64, GPU1,
XLA enabled and TF32 disabled. No NUTS route was used.

The controller-outcome.json file was saved before export and records
all_verified_members_exported=false. The later evidence/result.json is the
terminal report and records successful export and reload. The generic checkpoint
retains artifact_authority=false; it is not an independently admitted posterior
artifact. These files describe different stages of execution.

The source-copy manifest records commit 81f61a2c5 plus exact working-tree hashes.
The worker's later Git query records 7be915869 after the documentation commit.
The frozen source hashes identify executed code; the evolving shared checkout
does not. Earlier qualification-03 remains source-invalid and is not requalified.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this diagnostic as complete | Nine records retained; all six verified members exported and reloaded | Three candidate vetoes; no campaign execution failure | One target and supplied geometry | Preserve results and repair the remaining rejection-telemetry discrepancy | Release readiness or posterior correctness |
| Retain the six verified candidates | Fresh acceptance and health verification passed | No verification or reload veto for these members | Finite-horizon tuning does not establish convergence | Use within the stated diagnostic scope | Ranking or a new default |
| Preserve rejected-candidate evidence | Failed movement and numerical checks remain visible | One movement veto and two numerical vetoes | Modern telemetry does not establish the same safe-rejection invariants as the corrected TFP route | Inspect saved first-failure traces and test accepted-state/cache/MH-selection invariants | Evidence against HMC or permission to erase failures |

| Inference status | Finding |
|---|---|
| Hard veto screen | Three candidate failures under the frozen policy; six pass this diagnostic |
| Statistically supported ranking | None; ranking was outside the evidence contract |
| Descriptive-only differences | Runtime and acceptance summaries do not establish superiority |
| Default-readiness | Not established; posterior qualification and external-consumer admission remain false |
| Next evidence needed | Generic rejection-invariant tests, then fresh target-specific evidence for a changed kernel |

The strongest alternative explanation for the numerical rejections is the known
modern-versus-TFP rejection-telemetry mismatch, rather than corruption of retained
current states. Complete accepted-state/cache/MH checks could change their
interpretation in a future implementation; they cannot alter this frozen attempt.
The weakest evidence is downstream sampling: eight reload transitions provide
an engineering check only. An identity or source mismatch, broken endpoint
equality or omitted failed trial would overturn the engineering conclusion;
the terminal review found none.

## Budget and next work

The outer supervisor consumed 33,911.35313071398 inclusive seconds (9.42 hours),
including capacity waiting and all child stages. This is charged once to the
existing combined allocation. The released unused balance is
17,452.518772122326 seconds (4.85 hours); the separate SIR allowance of
7,363.064379415999 seconds is untouched. There is no active reservation and no
new numerical campaign was launched during closeout.

The next repair should inspect the saved failure traces and give the modern
randomized-HMC route explicit accepted-state, cached-value/score and MH-selection
checks consistent with the corrected TFP route. Numerical rejections may count
as zero acceptance only when those invariants are actually checked. Focused
finite/non-finite accepted/rejected tests must precede a fresh source-isolated
diagnostic. The current evidence and scientific policy must remain preserved.
This repair and broader posterior/external-consumer validation remain open;
the overall master program is not complete.

## Reproducibility

Artifact root: artifacts/modern-hmc-implementation-2026-10-07/N5/source-isolated-20261026-01.
Its manifest.json and qualification-01/evidence/manifest.json preserve the exact
commands, environment, data identity, device/memory settings, seeds and source
hashes. Qualification uses seed root (20261026,6101); reload uses its separately
declared namespace. Terminal evidence is qualification-01/evidence/result.json,
native-audit.json, members.json and reload/member-*/retained_archive.json.
closeout-review.json and settlement.json preserve the terminal checks and
budget reconciliation. The authoritative combined budget ledger remains
N5/diagnostic-extension-20261025-01/settlement.json.
