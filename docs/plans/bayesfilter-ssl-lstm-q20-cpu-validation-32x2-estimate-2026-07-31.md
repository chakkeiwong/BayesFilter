# q=20 CPU `32x2` Validation Estimate

Date: 2026-07-31
Status: `CONDITIONAL_PLANNING_ESTIMATE`

## Question

What happens to the previously measured q=20 campaign projection if each
64-row validation is evaluated by the CPU `32 workers x 2 rows` topology?

## Evidence

- CPU `32x2` standalone profile:
  `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/32x2-standalone-r1/result.json`
- Previous hybrid GPU projection:
  `docs/plans/artifacts/ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r2/projection.json`
- CPU warm validation-scale target call: `5.418130508662823 s` mean.
- CPU first validation-scale target call: `13.03513627499342 s`.
- Previous conservative GPU 64-row validation substitution:
  `741.1236495470075 s` per call.

## Calculation

The declared protocol has four tuning arms and two final streams. Assuming one
fresh CPU validation process per arm/stream, there are six first calls and 28
warm calls:

```text
CPU validation time = 6 * 13.03513627499342
                    + 28 * 5.418130508662823
                    = 229.91847189251956 s
                    = 3.831974531541993 min
```

The prior projection priced 34 GPU validation calls at
`34 * 741.1236495470075 = 25198.204084598256 s` (`6.9995 h`). Replacing that
component saves `24968.285612705736 s` (`6.9356 h`) before the planning margin.

Keeping the measured hybrid GPU optimizer and audit costs unchanged gives:

| Selected final architecture | Unbuffered conservative max | With existing 25% margin |
| --- | ---: | ---: |
| `(32,32)` | `1373045.4755724678 s` (`15.8917 d`) | `1716306.8444655847 s` (`19.8647 d`) |
| `(64,64)` | `1380450.0312243325 s` (`15.9774 d`) | `1725562.5390304157 s` (`19.9718 d`) |

The conservative architecture-max estimate is therefore approximately
`1,725,563 s`, or `479.3 h` (`19.97 days`). The previous architecture-max
estimate was `1,755,960 s`, or `487.8 h` (`20.32 days`), so the reduction is
approximately `30,398 s` (`8.44 h`).

## Assumptions And Limits

- Optimizer updates remain on the measured hybrid host-staged GPU backend.
- Untouched 256-row audits remain on that same GPU path.
- CPU validation uses one status-bearing value/score/status call per batch;
  no second CPU target call is priced.
- GPU-to-CPU row transfer, CPU pool startup, and status-wrapper overhead are
  not directly measured here; they must be measured before authorization.
- The current CPU pool profile returns value/score and parity-stable diagnostics,
  but a claim-bearing CPU validation replacement still needs status-preserving
  pool output and same-input CPU/GPU value/score/status parity.
- This estimate prices the already measured hybrid implementation. It is not a
  valid GPU-native campaign budget; the `tensorflow_eigh_strict` GPU-native
  route still requires parity and timing.

## Decision Table

| Decision | Primary criterion | Veto status | Next action | Nonclaim |
| --- | --- | --- | --- | --- |
| Use `32x2` as the validation candidate | Direct 64-row CPU profile completed with zero value/score drift | Status-preserving interface and cross-backend parity not yet checked | Implement/measure status-bearing CPU validation wrapper | No validation correctness or campaign authorization |
| Revise the old hybrid projection | Recompute all 34 validation calls with measured `32x2` timing | Remaining GPU update/audit costs dominate | Use `~19.97 d` only as a conditional planning number | No GPU-native budget claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | CPU profile finite, pinned, and parity-stable; replacement status/parity gate remains open |
| Statistically supported ranking | None; this is a deterministic timing estimate from one short profile |
| Descriptive-only difference | Validation component falls from about 7.0 h to 3.8 min under the stated assumptions |
| Default readiness | Not assessed |
| Next evidence needed | Status-bearing CPU pool, transfer-inclusive validation timing, and identical-input CPU/GPU parity |
