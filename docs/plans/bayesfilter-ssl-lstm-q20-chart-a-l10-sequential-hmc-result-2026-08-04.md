# q=20 Chart A L=10 Sequential Fixed-HMC Result

Date: 2026-08-04
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-chart-a-l10-sequential-hmc-plan-2026-08-04.md`
Status: `CANDIDATE_FAILED_SEQUENTIAL_ACCEPTANCE_GATE`

## Outcome

The frozen Chart A candidate `L=10`, step size `0.4148806556986277`, ran for
`33,585.7793 s` (`9.3294 h`) through three concurrent four-chain 500-draw
warm-up chunks. The third chunk failed the predeclared per-chain acceptance
gate:

```text
chain 0 mean acceptance probability = 0.22699457842828213
required interval = [0.35, 0.95]
```

The correct candidate decision is rejection under this sequential policy. The
run generated and archived 1,500 warm-up transitions per chain. Only the first
1,000 per chain preceded the veto and were policy-admissible warm-up progress;
the vetoing third chunk is preserved as failure evidence. The 2,000-transition
warm-up minimum was not reached, retained sampling never began, and R-hat and
ESS are unavailable.

All three chunks had finite required tensors, valid per-transition target
status, zero status-invalid rows, zero floor count, and movement in every
chain. Native divergence was not exposed by the installed TFP kernel and is
unavailable, not zero. Every chunk contained finite `abs(delta_h)` up to
`1e100`; this remains the prospectively declared explanatory alert and is not
the rejection reason.

After the shared controller had written the complete manifest, the launcher
hit a reporting-only `NameError` while calling the controller result's
`payload()` method: the shared worktree method referenced undefined
`SEQUENTIAL_NEUTRA_HMC_SCHEMA` instead of its defined archived schema constant.
This prevented `summary.json` and `sequential-result.json` but did not cause the
acceptance veto or alter the already-written samples, traces, receipts, or
manifest. The lane launcher now serializes the returned public dataclass fields
directly. The exact launch-time launcher was preserved as `source-at-launch.py`.

## Chunk Results

| Chunk | Slowest worker time | Acceptance probability by chain | Hard veto |
| ---: | ---: | --- | --- |
| `0` | `10,645.2750 s` | `(0.78270, 0.74072, 0.72679, 0.84350)` | None |
| `1` | `11,413.6790 s` | `(0.60974, 0.62906, 0.83873, 0.74517)` | None |
| `2` | `11,521.9741 s` | `(0.22699, 0.67212, 0.78518, 0.60930)` | `acceptance_probability_outside_declared_bounds` |

The acceptance changes are descriptive finite-run observations. The third
chunk's bound violation is a prospectively declared hard screen, not a
statistical ranking or evidence that the target or learned transport is wrong.

## Evidence Contract Result

| Evidence role | Result |
| --- | --- |
| Scientific question | Answered for this candidate: it did not remain inside the sequential kernel acceptance bounds. |
| Primary promotion criterion | Not reached; warm-up stopped before the 2,000-draw minimum. |
| Promotion veto | Fired: chain 0 chunk-2 acceptance `0.2269945784 < 0.35`. |
| Continuation veto | The candidate veto stopped sampling; later result serialization also failed, but only after complete archive creation. |
| Explanatory diagnostics | All chains moved; finite energy tails reached `1e100`; chunk times were 2.96-3.20 hours. |
| Not concluded | Posterior correctness, target invalidity, transport invalidity, NeuTra failure, Chart B behavior, sampler superiority, or default readiness. |

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the current `L=10`, step `0.4148806557` candidate | Warm-up minimum not reached | Acceptance veto fired in chain 0 | Whether a newly tuned smaller step repairs long-run acceptance | Return to fixed-HMC step repair/tuning under a new reviewed plan; do not continue this frozen kernel | Target, transport, or NeuTra invalidity |
| Preserve the three chunks as diagnostic evidence | Complete hashed samples/traces and manifest exist | No archive or finite/status veto | R-hat/ESS unavailable | Use only for kernel repair and runtime planning | Posterior estimates |
| Repair terminal serialization | Reporting method referenced an undefined symbol | Reporting-only engineering defect | Shared controller is concurrently modified | Keep lane-local serialization repair and add regression assertion | Any change to sampler verdict |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Supported: chunk 2 chain 0 acceptance fell below the declared bound. |
| Statistically supported ranking | None; one frozen candidate was tested. |
| Descriptive-only differences | Per-chain acceptance, runtime, movement, memory, and finite energy tails. |
| Default readiness | Not ready. |
| Next evidence needed | A newly tuned/repaired fixed-HMC step that passes fresh nomination, followed by a new full sequential campaign. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering | Four persistent CPU/XLA workers completed three chunks and archived all required tensors. Terminal result serialization failed after manifest creation. |
| Numerical/sampler | Required tensors and status passed; current kernel failed its acceptance gate before convergence diagnostics. |
| Scientific | No posterior or model conclusion is supported. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b370dc89e6e79f3853e0fccd5ab5b4fa2cb9065d` with unrelated dirty worktree preserved |
| Command | Exact command in `r1/launch.json` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; TensorFlow/TFP; FP64 |
| CPU/GPU | Four workers on CPUs `0..31`, supervisor on `32`; `CUDA_VISIBLE_DEVICES=-1`; no GPU used |
| XLA | Every worker emitted `Compiled cluster using XLA!` |
| Kernel | `L=10`, step `0.4148806556986277`, fixed identity `z` mass |
| Kernel hash | `34b89acd551dd25bee9dd0a463be67ff9d06f08ea3f970da5ffa97b44438ca4d` |
| Seeds | Controller seeds `(20260804,42010)`, `(20260804,43019)`, `(20260804,44028)` with recorded per-chain folds |
| Generated warm-up | 1,500 transitions per chain; third 500-draw chunk is veto evidence |
| Policy-admissible pre-veto warm-up | 1,000 transitions per chain |
| Retained draws | None |
| Controller wall through third chunk | `33,585.77929919 s` |
| Service terminal time | `2026-08-04T10:46:59+08:00` |
| Maximum worker RSS | `1,258,700,800 bytes` |
| Manifest | `docs/plans/artifacts/ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/r1/archive/chart-a-l10-manifest.json` |
| Manifest SHA-256 | `843b1231c567e030a7f1a7554576d157fcc56d3dac702f82866fc00c76ad2e48` |
| Preflight | `PREFLIGHT_PASSED`; SHA-256 `ad3a847ade89a0819dac573108d38f14a5863333573822d5e6b5f1808c472b4d` |
| Tests before launch | `20 passed` |

## Negative-Result Classification

- Implementation failure: terminal result serialization only; sampling and
  archive completion were not invalidated.
- Tuning failure: supported for the frozen `L=10` step under the declared
  sequential acceptance policy.
- Diagnostic failure: R-hat/ESS were correctly unavailable because warm-up
  stopped before their minimum.
- Evidence against the scientific idea: none. A smaller freshly tuned step or
  another nominated fixed-HMC candidate may remain viable.

## Post-Run Red Team

The strongest alternative explanation is finite-run acceptance variability:
the first two chunks passed and the third chain-0 rate fell sharply. That does
not erase the prospectively declared veto; it does limit generalization beyond
the current frozen candidate. A new step nominated under a reviewed repair
plan and then passing a full independent sequential run would overturn the
candidate-level rejection. The weakest evidence is the absence of R-hat, ESS,
retained posterior draws, and native divergence telemetry.
