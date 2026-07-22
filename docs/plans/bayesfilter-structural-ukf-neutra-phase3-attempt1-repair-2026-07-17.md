# Structural UKF NeuTra Phase 3 Attempt 1 and Repair

Date: 2026-07-17

Status: `ATTEMPT_1_TUNING_REJECTED_REPAIR_AUDITED_READY`

## Attempt 1 result

The initial GPU/XLA confirmation attempt completed without an infrastructure,
target-identity, serialization, or device failure. It exhausted the six
predeclared fixed-kernel candidates and issued
`SAMPLER_BLOCKED_NO_TUNING_ADMISSION`. No warm-up, retained posterior sample,
or truth-tail result was produced, so this is a tuning failure and not evidence
against the structural posterior or the NeuTra scientific direction.

| Step size | Leapfrogs | Acceptance | Energy-screen failures | Max modern R-hat | Admission |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.20 | 8 | 0.96125 | 2 | 1.00526 | reject: health veto |
| 0.30 | 8 | 0.91325 | 1 | 1.01147 | reject: health and R-hat |
| 0.10 | 8 | 0.74575 | 74 | 2.53176 | reject: health and R-hat |
| 0.40 | 8 | 0.82825 | 3 | 1.03853 | reject: health and R-hat |
| 0.05 | 8 | 0.99475 | 0 | 1.04044 | reject under initial 1.01 tuning gate |
| 0.02 | 8 | 0.98300 | 4 | 1.58987 | reject: health and R-hat |

Binding Attempt 1 result:

- path: `docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/confirmation/attempt-01/result.json`;
- SHA-256: `622ab339ceffdeb4850b53d9930a372dd9ea9e4ab13a7c59b4401bde10c8ffbe`;
- wall time: `12057.44` seconds;
- target signature:
  `e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`.

## Repair question and evidence contract

Question: can the only energy-stable verified kernel, step size `0.05` with
eight leapfrog steps, enter adaptive warm-up under the already declared warm-up
R-hat gate and subsequently satisfy the unchanged retained and truth-tail
criteria?

The repair changes tuning admission from modern R-hat `<=1.01` to `<=1.05`.
This fixes a stage-role mismatch: `1.05` is the declared warm-up sufficiency
gate, while `1.01` is the final retained-posterior gate. Requiring `1.01`
before starting adaptive warm-up made the tuning stage stricter than the
warm-up stage it was intended to enter. The repair does not relax final
promotion.

- Candidate: step size `0.05`, eight leapfrog steps, trajectory length `0.4`.
- Tuning admission: finite states and log density, valid target telemetry, zero
  energy-screen failures, and modern R-hat `<=1.05` on a fresh 1,000 burn-in
  plus 1,000 verification draws per chain.
- Warm-up: fresh seed, archived and excluded from posterior summaries, modern
  R-hat `<=1.05`, 2,000 minimum and 10,000 maximum per chain.
- Retained promotion: unchanged modern R-hat `<=1.01`, minimum bulk ESS
  `>=1000`, minimum tail ESS `>=400`, zero hard health vetoes, 4,000 minimum
  and 10,000 maximum per chain.
- Scientific promotion: unchanged all-five-parameter posterior truth-tail
  `p_truth>=0.05`; marginal and severe rules remain unchanged.
- Artifact: fresh `confirmation/attempt-02/`; Attempt 1 is never overwritten.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Step size 0.05 | only Attempt 1 verification with zero energy failures | moves too slowly | fresh disjoint tuning and adaptive retained ESS | evidence-selected repair |
| Eight leapfrogs | Attempt 1 trajectory was stable at 0.05 | trajectory too short for retained ESS | retained ladder to 10,000 | preserved baseline |
| Tuning R-hat 1.05 | predeclared warm-up gate and standard sequential policy | admits a kernel that never reaches final convergence | final 1.01 plus ESS gates remain mandatory | stage-corrected gate |
| Fresh seeds | prevents reusing favorable Attempt 1 randomness | fresh run may fail | exact manifest and archive hashes | required repair evidence |

## Skeptical repair audit

Verdict: `PASS`.

- Wrong baseline: none is introduced; the same frozen transport and posterior
  target are used.
- Proxy risk: tuning admission only permits warm-up. It cannot establish the
  retained posterior or truth-tail result.
- Missing stop conditions: energy health, warm-up cap, retained cap, truth-tail
  rules, eight-hour campaign wall budget, and the Phase 3 repair limit remain.
- Hidden assumption: the repair assumes longer sequential accumulation can
  overcome the observed stable-kernel autocorrelation; retained ESS and modern
  R-hat directly test this assumption.
- Environment and artifact risk: GPU/XLA, memory growth, fresh seeds, parent
  result binding, and unique output root are checked in the harness.
- Misleading-success risk: passing tuning alone is explicitly insufficient;
  only the unchanged retained and truth-tail gates can produce the requested
  structural result.

Attempt 2 may proceed. If tuning, warm-up, or retained sampling fails, do not
interpret truth recovery. Record the sampler blocker and decide whether the
remaining one localized Phase 3 repair should address trajectory length or
transport quality.
