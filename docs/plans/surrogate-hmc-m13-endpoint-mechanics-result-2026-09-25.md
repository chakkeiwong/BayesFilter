# M13 surrogate-HMC mechanics result — 2026-09-25

The master executed 24 transitions (8 warm-up, 16 retained) on calibration
seed 1704 using the canonical analytical FP32/TF32/XLA DCTR proposal force
and a separate fixed-seed importance endpoint. All transitions were finite,
all endpoints valid, and no force fallback or divergence occurred. All 24
proposals were accepted; their mean acceptance probability was 0.989427745.
This establishes a short mechanics result, not posterior convergence.

The endpoint computes the full-support range/bearing likelihood by 4096-particle,
8-replicate FP64 defensive importance estimation with common random numbers,
then adds the exact independent N(0,log(2)^2) prior. The value is rounded to
FP32 for the FP32 HMC integrator and acceptance calculation. It is a fixed
finite approximation to the model likelihood; its uncertainty and posterior
effect are not yet qualified. The separately computed canonical force is
`-score_DCTR + theta/log(2)^2`.

Audit of attempt03 found a scope-label mismatch and an eager enclosing HMC
loop. Both were repaired before attempt04: the entire shared HMC loop now uses
a stable-signature XLA function, and the control-family label describes the
actual geometric protection controls. All energy and call-count traces are
preserved. Attempt04 passed all numerical checks; its 24 transitions used
24 endpoint evaluations and 72 force evaluations. The initial endpoint was
computed separately. No target controls were changed by this repair.
The old 129x129 posterior grid uses calibration seed 1701, so it cannot validate
the seed-1704 posterior in this run.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Short mechanics passed | All 24 transitions finite | No fallback, divergence, or endpoint failure | Endpoint approximation; untuned controls and short chain | Assess same-data endpoint uncertainty on actual proposal pairs | Posterior correctness, convergence, control selection |

| Inference status | Finding |
|---|---|
| Hard veto screen | All declared numerical checks passed |
| Statistically supported ranking | None; no competing sampler was run |
| Descriptive-only differences | 100% observed acceptance; 98.94% mean acceptance probability |
| Default-readiness | Not established; endpoint and posterior qualifications remain |
| Next evidence needed | Same-data endpoint uncertainty, scoped controls, public HMC tuning and retained posterior comparisons |

Attempt -01 failed on scalar instead of batched endpoint potential; repaired
with a regression. Attempt -02 passed 51 tests but could not reserve the full
worker GPU allowance after retaining -01 charges. The pass cap was adjusted
from 1200 to 1400 GPU seconds within the unchanged global campaign budget.
Attempt -03 passed the same 51 tests and completed both jobs. After the full-loop
repair, attempt -04 passed 52 tests and completed the compiled mechanics check.
Its internal pass cap was 1600 CPU / 2800 GPU seconds, including prior charges. No prior evidence
was overwritten. This is localized harness repair, not a failure of surrogate HMC.

Command:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-endpoint-mechanics --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-mechanics-04
```

Evidence: `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-mechanics-04/m13-endpoint-mechanics/report.json`.
The parent manifests record commit 8a4cffa2fd085f43ab5710c24d6e7b48770aea0d,
source snapshots, commands and verified GPU memory growth. Successful attempt:
267.604591 CPU worker seconds and 163.019875 GPU seconds. Cumulative campaign:
46939.120100 CPU / 36164.030744 GPU seconds; remaining 7060.879900 CPU /
190635.969256 GPU seconds. Budget limits remain 54000 CPU / 226800 GPU seconds.

The strongest alternative explanation for high acceptance is simply the very
small step size and two leapfrog steps. It says nothing about exploration. A
long, properly tuned comparison against a qualified same-data posterior would
be required to assess that question. The next actions are inside the existing
master; no Claude review or LG1 rerun is requested.
