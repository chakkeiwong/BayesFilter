# q=20 CPU-XLA 32x1 Distributed NeuTra-HMC Validation

> Superseded 2026-08-02 for claim-bearing tuning: the historical launcher used
> a custom grid instead of `tune_fixed_transport_hmc_kernel` and treated a
> finite log-accept energy proxy as a hard veto. Its mechanics-only preflight
> remains diagnostic; its tuner must not be rerun.

Date: 2026-08-01
Tier: serious local CPU/XLA research campaign
Status: `READY_TO_EXECUTE`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Main question | Do the two newly trained q=20 `(32,32)` NeuTra charts admit finite, moving fixed HMC kernels and pass modern multi-chain warm-up and retained-sample diagnostics when executed as 32 independent CPU/XLA chains? |
| Exact inputs | Seed A checkpoint program step `1,500` with selected best trainer state step `1,500`, and Seed B checkpoint program step `2,500` with controller-selected best trainer state step `2,250` (the checkpoint's controller best-step selection is `2,500`) from `ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1`. The selected state hashes are recorded in every worker receipt. |
| Topology | 32 one-chain workers on CPUs `0..31`, allocated as 16 independent chains per chart; supervisor pinned exclusively to CPU `32`. |
| Exact comparator/baseline | Identity-mass fixed HMC in each exact learned chart. The mechanics-canary step `0.01` is not a kernel candidate and cannot be promoted. |
| Tuning criterion | For each chart, select prospectively from `L={2,4}` and step sizes `{0.4,0.5656854249492381,0.75,1.0}`. Two independent 32-transition replications must be finite/moving with pooled mean Metropolis acceptance probability in `[0.55,0.85]`. The deterministic representative minimizes distance to `0.70`, then smaller `L`, then smaller step. A 16-chain 64-transition confirmation must have every-chain mean acceptance probability in `[0.35,0.95]`, finite telemetry, and movement in all chains. |
| Warm-up gate | After `2,000` warm-up transitions per chain, the latest `1,000` transitions across the 16 chains have finite maximum rank/folded R-hat `<=1.05` in both NeuTra and mapped model coordinates. If not ready, continue in 500-transition chunks to a maximum `10,000`. |
| Retained gate | At cumulative retained checkpoints from `1,000` to `10,000` transitions per chain, both coordinate systems must have finite maximum rank/folded R-hat `<=1.01`, minimum bulk ESS `>=400`, and minimum tail ESS `>=400`. |
| Hard vetoes | Checkpoint/hash/target/transport mismatch; visible GPU; non-XLA execution; nonfinite state, score, target, or log acceptance; invalid target status on audited states; unmoved chain in a chunk; exposed positive native divergence; `abs(log_accept_ratio)>1000`; archive/hash failure; worker crash; affinity drift; aggregate worker RSS over 64 GiB; or campaign cap. |
| Explanatory only | Acceptance, timing, cold compile, RSS, per-chart differences, continuous R-hat/ESS before a pass, and native divergence unavailability. |
| Repair trigger | A chart that has no confirmed kernel or reaches a warm-up/retained cap without passing is a kernel/transport repair trigger. It does not reject the other chart or NeuTra generally. |
| Continuation veto | Shared target/checkpoint/transport invalidity, shared worker failure, source drift, archive invalidity, resource cap, or missing required diagnostics. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-2026-08-01/r1/` with tuning, worker manifests, per-chain chunk tensors, diagnostics, progress, and terminal summary. |
| Nonclaims | No posterior oracle, stationarity proof, model adequacy, predictive validity, chart ranking, CPU default, GPU equivalence, or broad scientific validity. |

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
| --- | --- | --- | --- |
| 16 chains per chart | User-authorized 32-worker topology; campaign hypothesis | Uses every worker and gives more replicated starts than the four-chain minimum | Extra chains do not shorten within-chain transitions; warm-up/retained diagnostics and wall time expose value/cost |
| Identity mass | Existing q=20 NeuTra-HMC baseline | Tests whether the learned chart supplies usable geometry | Residual anisotropy; tuning miss or R-hat/ESS cap triggers metric/transport repair |
| Step grid | Inherited old q=20 midpoint plus adjacent target-specific hypotheses | Covers the prior viable neighborhood without using canary `0.01` | May omit a viable kernel; no confirmed candidate yields tuning failure, not relaxed gates |
| `L={2,4}` | Old q=20 pilots found both viable for chart A; target-specific warm start | Limits cost while testing short trajectories | Longer trajectory may be required; failure triggers a new reviewed tuning scope |
| Tuning acceptance bands | Reviewed starting hypotheses, deliberately broader than the old brittle every-chain pilot | Mean Metropolis acceptance is a tuning quantity, not convergence evidence | Over-broad band admits inefficient kernel; long warm-up R-hat/ESS remains decisive |
| 500-transition chunks | Repository sequential policy convenience choice | Matches diagnostic cadence and keeps artifacts bounded | Fixed graph cost and archive overhead; first chunk records measured rate |
| Warm-up/retained thresholds | Repository `bayesfilter_neutra_sequential_hmc_v1` policy | Required for serious NeuTra HMC | Finite-sample screens are not posterior truth; nonclaims remain explicit |
| CPU/XLA | User-selected exception after measured p32 canary | 32 workers achieved `24.56x` descriptive throughput with finite mechanics | Repository default is GPU; manifest records CPU diagnostic/validation exception |

## Numerical Provenance

- `32+1` cores: user selected; measured feasible in the p32 canary.
- `16` chains/chart: derived equal allocation of 32 workers to two charts;
  hypothesis, not reviewed universal default.
- `2,000/1,000/10,000`, R-hat `1.05/1.01`: repository policy.
- ESS `400`: inherited confirmation threshold in the existing q=20 sequential
  plan; reviewed starting criterion, not a theorem.
- Tuning grid and bands: target-specific hypotheses based on the previous q=20
  tuning neighborhood; not promoted settings.
- Campaign cap `86,400 s`: convenience ceiling derived to cover the p32 short-
  call estimate (`5.76 h` minimum) plus tuning, compilation, chunk diagnostics,
  and a substantial uncertainty margin. It is a cap, not expected use.

## Skeptical Pre-Execution Audit

- Wrong baseline: repaired. The prior canary step `0.01` was mechanics-only and
  is excluded; both charts tune exact new payloads from scratch.
- Proxy promotion: no. Tuning acceptance only admits a fixed-kernel candidate;
  retained R-hat/ESS decides sampler admission.
- Missing stop: no. Finite tuning grid, 64-transition confirmation, warm-up and
  retained caps of `10,000`, per-chunk vetoes, 64 GiB RSS veto, and 24-hour wall
  cap are explicit.
- Unfair chart comparison: charts are replications, not ranked. They tune
  separately and use disjoint chains/seeds.
- Stale context: old q=20 tuning artifacts bind different payload hashes and
  are warm-start context only.
- Environment mismatch: every worker hides CUDA before TensorFlow import,
  verifies no GPU, uses one physical core, one TF/BLAS thread, FP64, and XLA.
- Artifact adequacy: each chain writes separate warm-up and retained chunk
  tensors plus trace summaries; warm-up never enters retained diagnostics.
- Misleading pass: R-hat/ESS can pass without posterior correctness; both
  coordinate systems are checked and nonclaims remain explicit.
- Misleading failure: a fixed identity-mass kernel may fail despite a viable
  transport; failure is classified as a kernel/transport repair trigger.
- Research guardian: a failed chart does not invalidate the target, harness,
  or research direction unless a shared continuation veto fires.

Audit decision: `PASS_FOR_FRESH_TUNING_THEN_DISTRIBUTED_SEQUENTIAL_VALIDATION`.

Implementation audit addendum, 2026-08-01: the initial unexecuted launcher was
not adequate because it ran the charts serially, used one tuning replication,
rebuilt chunk graphs, lacked several declared vetoes, and did not write raw
chunk archives. Before execution it was repaired to run chart A and B
concurrently on their disjoint 16-worker groups, use two independent tuning
replications per arm, cache static XLA runners, audit final-state target status,
enforce all declared worker/RSS/affinity/finite/divergence/log-accept vetoes,
use R-hat-only warm-up readiness, skip retained work after warm-up failure, and
write hashed per-chain tensor shards. A full-topology two-transition step-0.01
preflight is mechanics-only and cannot contribute tuning or validation evidence.
The preflight also verified that the Seed B file named `checkpoint-2500.json`
contains controller-selected best-step metadata `2500` while its selected
trainer-state payload has internal optimizer step `2250`; these are distinct
fields and are recorded separately, not silently relabeled.

## Execution Sequence

1. Validate both checkpoints, restore best trainer states, freeze/reload exact
   transport payloads, and verify target/transport identity inside every worker.
2. Start 16 persistent chain workers per chart under the 32+1 topology.
3. Run the fresh bounded tuning grid and 16-chain confirmation concurrently for
   both charts; each tuning arm has two independent replications.
4. Stop any chart without a confirmed kernel. Do not spend long sampling budget
   on an unconfirmed kernel.
5. For each confirmed chart, execute 500-transition warm-up chunks; archive all
   draws but exclude them from posterior estimates. Apply chunk hard vetoes.
6. At/after 2,000 warm-up transitions, evaluate the latest 1,000 transitions in
   both coordinate systems. Continue to 10,000 only if needed.
7. Once warm-up passes, reset retained buffers and run 500-transition retained
   chunks. Check both coordinate systems at/after 1,000; stop on pass or at
   10,000.
8. Write a terminal summary and result note separating engineering, sampler,
   and scientific ledgers.

## Command

```bash
taskset -c 32 timeout 86400 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_cpu_xla_32x1_distributed_hmc_validation_2026_08_01.py \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-2026-08-01/r1
```
