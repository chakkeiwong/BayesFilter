# SSL-LSTM q=20 Seed-B Terminal NeuTra Validation Result

Date: 2026-08-07  
Status: `SEQUENTIAL_SCREEN_PASSED`

## Outcome

The clean seed-B terminal NeuTra transport supports a viable fixed-HMC sampler
under the repository sequential policy. A fresh seed-B-bound six-`L` tuning
grid nominated `L=3`, step size `0.8115211181271775`, with fixed identity mass
in NeuTra coordinates. The repaired sequential run then passed at the policy
minimum:

- four chains;
- 2,000 discarded warm-up transitions per chain;
- latest warm-up window maximum rank/folded R-hat `1.0070048645695828`;
- 1,000 retained transitions per chain, 4,000 retained draws total;
- retained maximum rank/folded R-hat `1.007235512713807`;
- retained minimum bulk ESS `1380.0879940941147`;
- retained minimum tail ESS `533.6942079672039`;
- zero recorded target-status-invalid transitions;
- every chain moved in every chunk;
- every per-chain chunk acceptance probability remained within `[0.35,0.95]`;
- all required state, target, proposed-target, score, log-acceptance, and energy
  tensors were finite; and
- no declared hard veto fired.

This answers the campaign question positively at the sampler-screen level:
seed B produced a properly trained NeuTra candidate worth posterior validation.
It does not establish that the retained distribution equals a trusted posterior
or that the model is scientifically adequate.

## Tuning Result

The six-`L` tuning phase completed in `28222.05524148399` seconds. Only `L=3`
passed the public tuner's screen and fresh verification. Its kernel was:

| Field | Value |
| --- | ---: |
| Leapfrog steps | `3` |
| Step size | `0.8115211181271775` |
| Fresh verification mean acceptance probability | `0.6881377960324893` |
| Fresh verification binary acceptance rate | `0.69921875` |
| Kernel hash | `76729f30a8a9ed90966db955006496537eb784638d968e9e6beeaad664ca7ea3` |
| Mass | Fixed identity in NeuTra coordinates |
| XLA | Enabled |

The other five arms did not pass their fresh verification or did not nominate a
step. These are per-arm tuning outcomes, not a statistically supported ranking
of trajectory lengths.

## Sequential Diagnostics

| Diagnostic | NeuTra coordinates | Model coordinates | Gate |
| --- | ---: | ---: | ---: |
| Warm-up maximum R-hat | `1.0060574523845958` | `1.0070048645695828` | `<=1.05` |
| Warm-up minimum bulk ESS | `3062.1712500647254` | `3507.6741072777104` | Explanatory |
| Warm-up minimum tail ESS | `1061.1498300830935` | `1293.178423484194` | Explanatory |
| Retained maximum R-hat | `1.00672338795167` | `1.007235512713807` | `<=1.01` |
| Retained minimum bulk ESS | `1441.1683946220742` | `1380.0879940941147` | `>=400` |
| Retained minimum tail ESS | `533.6942079672039` | `580.385724625388` | `>=400` |

Four 500-transition warm-up chunks and two 500-transition retained chunks were
archived. Warm-up draws were excluded from retained diagnostics. The sequential
wall time was `25231.886756150023` seconds, about 7.01 hours. Chunk times ranged
from about 4,183 to 4,238 seconds.

Acceptance probabilities across the six chunks ranged from
`0.6638980601540271` to `0.7612168992473286`. Acceptance passed its declared
bound but is not convergence evidence.

## Numerical And Status Audit

All 54 archived sample/trace tensor receipts were independently rehashed after
the run; every SHA-256 matched. Across six chunks, all four chains moved, all
target-status counts were zero, all acceptance values were within the declared
bound, and every chunk recorded `hard_vetoes=[]`.

TFP does not expose a native divergence flag for this kernel, so native
divergence status remains `not_exposed_by_kernel`, not zero. The finite
`abs(delta_h)` tail was very large, including the target sentinel magnitude in
some chunks. Under the prospective contract this is an explanatory alert, not
a standalone veto; required tensors remained finite, target status passed, and
the sequential convergence/ESS gates passed.

TensorFlow emitted complex128-to-float64 cast warnings while computing
ESS/R-hat. The diagnostic implementation calls TFP's effective-sample-size
routine on real float64 archived inputs; TFP's spectral calculation uses a
complex intermediate and returns real-valued diagnostics. The resulting R-hat
and ESS tensors were finite. This warning is recorded as residual numerical
diagnostic risk, not silently treated as proof of validity or as a hard veto.

## Localized Harness Repair

The first post-tuning preflight stopped before sampling because the sequential
wrapper used a different transformed-target scope from the kernel's exact
tuning scope. Since scope is part of the adapter signature, fail-closed identity
checking correctly rejected it. The wrapper was repaired to preserve the exact
tuning scope. No checkpoint, target, transport, kernel, random seed, sampler,
threshold, or hardware class changed. Twenty-six focused tests and a fresh real
preflight passed, then only the previously unstarted sequential phase was run.

This was an implementation integration failure and repair. It was not a failed
NeuTra, HMC, target, or numerical result.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit seed B as a viable NeuTra/fixed-HMC candidate | Passed sequential R-hat and ESS gates at 2,000 warm-up plus 1,000 retained transitions per chain | No declared hard veto; target-invalid count zero; native divergence unavailable | No trusted posterior/reference agreement check | Run an untouched posterior/reference validation using these retained draws and a prospectively declared comparator | Posterior correctness, model adequacy, superiority, robustness, or default readiness |
| Preserve `L=3`, step `0.8115211181271775` for this exact candidate | Fresh seed-B tuning and sequential screen passed | No kernel/status veto | One candidate and one sequential random-stream bank | Freeze kernel for the posterior-validation phase; do not retune on retained draws | Universal q=20 kernel or statistical superiority over other `L` values |
| Do not generalize seed-A invalidity to NeuTra | Seed B completed clean training and sequential sampling | Seed A retains its own target-validity promotion veto | Cross-seed robustness remains unknown | Treat seed A and B as different candidate outcomes | General NeuTra reliability across training seeds |

## Inference Status

| Evidence class | Result |
| --- | --- |
| Hard veto screen | Passed for the seed-B tuned kernel and all six sequential chunks |
| Viable candidates | Seed-B terminal checkpoint 4000 with tuned `L=3` kernel |
| Statistically supported ranking | None; the campaign did not support a stochastic ranking across kernels, seeds, or methods |
| Descriptive-only differences | Tuning-arm acceptance/runtime, chunk acceptance, continuous R-hat/ESS margins, and energy tails |
| Default readiness | Not evaluated and not established |
| Next evidence needed | Untouched posterior/reference agreement and downstream model-specific validation |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Passed focused tests, exact checkpoint/kernel/adapter bindings, CPU affinity, GPU hiding, XLA receipts, immutable chunk archives, and post-run hash verification |
| Numerical/sampler validity | Passed declared target-status, finite-tensor, movement, acceptance-bound, R-hat, and ESS screens; native divergence unavailable; large finite energy tails and TFP complex-cast warnings remain recorded limitations |
| Scientific interpretation | A viable sampler candidate exists. Posterior correctness and scientific/model validity remain untested |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit at initial campaign launch | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` |
| Worktree | Dirty with this lane and concurrent agents' work; unrelated files were preserved |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, Python 3.13, TensorFlow/TFP CPU/XLA FP64 |
| GPU status | `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import in supervisor and every worker; no GPU used |
| Tuning topology | Six concurrent arms, 58 worker cores plus CPU 127 supervisor |
| Sequential topology | Four persistent one-chain workers, eight cores each, CPU 32 supervisor |
| Training checkpoint | Seed-B continuation update 4000, optimizer step 6250, SHA-256 `f70546b04094b9c838382daddf3bdbbdfd6501e5c5f119e7cb80f4a0a954b32d` |
| Tuning wall | `28222.05524148399 s` |
| Sequential wall | `25231.886756150023 s` |
| Approximate successful tuning plus sequential wall | `53453.941997634014 s` (`14.85 h`) |
| Random seeds | Fresh 2026-08-07 tuning/screen/verification streams; sequential root `(20260807,41001)` |
| Tuning artifact SHA-256 | `2a50228966aec9c2dc30d897fb8c7c1b8e649bb4558ea9adf9ce6bf7747d0a7f` |
| Sequential summary SHA-256 | `279bfbdfa244dcba28ec63c6cc1168be0273bb3558c70a70a1eae701b7e73165` |
| Sequential result SHA-256 | `fb84554ab335d53a16af7b252050bd011e2e67470dc6e77eb38332b569ccec94` |
| Archive manifest SHA-256 | `9ed6735e7bb2127b157570a98849156acfda92d2e32aeed87a253506ae69c20b` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-terminal-neutra-validation-plan-2026-08-07.md` |
| Result | This file |

## Post-Run Red Team

Strongest alternative explanation: four chains initialized in a modest NeuTra
region can agree on the same local mode while missing another important mode or
while sampling a numerically consistent but scientifically wrong target. This
is why the next phase must compare posterior summaries against an untouched
reference or another independently justified posterior authority.

What would overturn the bounded conclusion: a verified archive hash mismatch,
an independent diagnostic showing the reported R-hat/ESS calculation is wrong,
a target-status audit identifying invalid retained states, or a replicated
sequential run that fails the same prospective gates. No such evidence was
observed here.

Weakest evidence: only one trained seed-B transport and one sequential random
stream were admitted, and retained sampling stopped at the 1,000-per-chain
minimum. The result supports viability, not robustness or superiority.

