# Structural UKF NeuTra Phase 3 Attempt 2 and Final Repair

Date: 2026-07-17

Status: `ATTEMPT_2_TUNING_REJECTED_FINAL_REPAIR_AUDITED_READY`

## Attempt 2 result

Attempt 2 tested the only energy-stable Attempt 1 kernel on fresh seeds and
used the stage-correct modern R-hat tuning threshold of `1.05`. The kernel
again had zero energy-screen failures and all target telemetry was valid, but
the fresh disjoint verification reached max modern R-hat `1.07895`, so it was
correctly rejected. No warm-up, retained posterior, or truth-tail result was
produced.

Binding result:

- path: `docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/confirmation/attempt-02/result.json`;
- SHA-256: `b02d22b9cfa51995c02c0df3f686770660d99a1f461600d224134b32ee200eed`;
- step size/leapfrogs/trajectory: `0.05 / 8 / 0.4`;
- acceptance: `0.99925`;
- energy-screen failures: `0`;
- per-parameter modern R-hat:
  `[1.02811, 1.07895, 1.05830, 1.06566, 1.04830]`;
- wall time: `2054.22` seconds.

The result confirms numerical stability at step size `0.05`, but the high
acceptance and failed R-hat indicate that trajectory length `0.4` moves too
little on this seed. This is a mixing/tuning failure, not a structural
truth-tail result.

## Final repair contract

The final allowed localized repair keeps the stable step size `0.05` and
increases leapfrog steps from 8 to 12, giving trajectory length `0.6`. It uses
fresh probe, verification, warm-up, and retained seeds and a fresh
`confirmation/attempt-03/` output root.

- Tuning admission remains modern R-hat `<=1.05`, zero energy-screen failures,
  finite values, and valid target telemetry after 1,000 burn-in plus 1,000
  verification draws per chain.
- Warm-up remains adaptive from 2,000 to 10,000 archived draws per chain with
  modern R-hat `<=1.05`; warm-up is excluded from posterior summaries.
- Retained promotion remains modern R-hat `<=1.01`, minimum bulk ESS `>=1000`,
  minimum tail ESS `>=400`, zero hard health vetoes, and 4,000 to 10,000 draws
  per chain.
- Truth-tail pass/marginal/severe thresholds remain `0.05` and `0.003` exactly.
- The frozen transport, target identity, data, model, hardware class, and
  claim scope do not change.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Step size 0.05 | zero energy failures in both completed verifications | still encounters rare unstable region | fresh health gate | evidence-supported stable scale |
| Twelve leapfrogs | 1.5 times the failed trajectory, bounded by remaining campaign budget | still too short or newly unstable | fresh tuning verification | repair hypothesis |
| Singleton candidate | isolates the diagnosed trajectory mechanism and avoids another broad grid | misses a better pair | explicit nonclaim and terminal blocker if it fails | cost-bounded choice |
| Fresh seeds | prevents favorable-seed reuse | fresh result differs | exact manifest | required evidence |

## Skeptical audit

Verdict: `PASS`.

- The repair targets the observed failure directly: stable integration but
  insufficient mixing.
- It does not promote acceptance or a training proxy into a scientific gate.
- It does not relax the energy, modern R-hat, ESS, truth-tail, draw-cap, or
  artifact requirements.
- At measured throughput, probe, verification, minimum warm-up, and minimum
  retained sampling should fit inside the remaining eight-hour HMC campaign
  budget. Any cap extension or further Phase 3 retry requires new direction.
- A successful tuning command alone remains insufficient. A structural NeuTra
  pass requires valid retained HMC and the declared truth-tail result.

Attempt 3 may proceed. If it fails tuning, warm-up, retained convergence, or
health, close the structural test as sampler-inconclusive under this bounded
campaign and document the exact blocker rather than launching another repair.
