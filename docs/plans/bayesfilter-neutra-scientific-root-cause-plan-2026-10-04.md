# Root-cause investigation of the corrected NeuTra mixture fits

Question: why do the saved r3 IAF fits fail even with exact iid teachers, and
why can the reverse-KL finish reduce its own objective while damaging coverage?
This investigation evaluates the eight existing forward/final checkpoints. It
does not train new maps, change the canonical architecture, or evaluate fresh
generalization targets. The requested code-and-math explanation authorizes this
bounded reference diagnostic.

## Evidence contract and intent

- Baselines: each fit's own frozen forward endpoint versus its RKL endpoint;
  the normalized mixture and its exact responsibility-moment identities are
  the reference. Original r1 failure labels are invalid and excluded.
- Mechanisms: density/inverse/gradient error; failure to fit local component
  shape and low-density regions; optimization that has not settled; the change
  of objective from forward to reverse KL; conditional-scale saturation.
- Primary outcome: localize observed discrepancies using analytic reference
  moments and quantify paired endpoint changes. This is diagnosis, not a new
  map-admission or method-ranking experiment.
- Continuation veto: wrong snapshot/checkpoint identity, nonfinite evaluation,
  failed density/derivative invariant, exhausted diagnostic allocation. An
  invariant failure suspends geometric interpretation until localized.
- Repair triggers: materially wrong derivatives or inverse; residual local
  moments, excess mass outside all component ellipses; RKL coverage regression.
- Explanatory only: training/heldout loss gap, Jacobian singular values, score
  residuals, cap slopes, conditional KL decomposition, learning histories.
- Limits: finite checks cannot prove global correctness, optimal training,
  representational impossibility, posterior/HMC readiness, or six-method/q20
  transfer. No new default or ranking of training recipes will be declared.

For a mixture p=sum_k w_k phi_k and responsibilities rho_k=w_k phi_k/p,
E_p[rho_k]=w_k, E_p[rho_k y_k]=0, E_p[rho_k y_k y_k^T]=w_k I.
Here y_k=(x-mu_k)/sqrt(v_k). In two dimensions
E_p[rho_k 1(||y_k||^2>8)]=w_k exp(-4). These identities remove the noisy
reference mean from the diagnostic comparison. The probability outside the
union of these ellipses is at most exp(-4), by conditioning on the true
component. This is a reference bound, not a newly tuned admission threshold.

## Execution and numerical choices

1. Trace the actual executed snapshot and compare key live files. Inspect the
   author mask/conditional-cap source alongside the local equations.
2. Reload every saved checkpoint. Check inverse round trips, analytic log
   determinant against a full 2D autodiff Jacobian and finite differences,
   analytic mixture scores, and the manual pullback against autodiff. Check
   forward-KL parameter directional derivatives and standard RKL's stopped-score
   gradient against differentiation through the actual target.
3. Evaluate saved train/holdout clouds, and 32,768 fresh common latent draws
   per target. Report Monte Carlo standard errors conditional on each fixed
   map, paired endpoint changes, exact responsibility moments, out-of-ellipse
   mass, and the exact augmented-distribution KL decomposition. These standard
   errors describe integration, not variation across training seeds.
4. Plot the two width-64 fitted densities before/after RKL beside their exact
   targets. A finite grid illustrates geometry; it is not integration evidence.
5. Write the mathematical root-cause report, decision/inference tables and reset
   memo update, including unresolved capacity/optimizer hypotheses.

CPU-only FP64/no-XLA is an explicit independent diagnostic exception, using
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. No optimizer updates or
training occur. Repeated numerical functions have stable TensorFlow signatures;
no pfor is used. Existing GPU/XLA campaign evidence remains the training record.

Numbers and provenance:

| Choice | Provenance / reason | Failure mode / early check | Status |
|---|---|---|---|
| Eight r3 checkpoints | All four completed fits, both endpoints | Selection bias: include all | Fixed evidence |
| 32,768 latent draws | Convenience power-of-two; bounded Bernoulli SE <=0.00277 | Rare tails remain noisy; report SE, do not certify tails | Diagnostic hypothesis |
| Seeds [20261004,9101/9201] | Fresh diagnostic date plus saved target seed; common draws across endpoints | Not fit-seed replication | Convenience |
| 64 gradient-check rows | Small batch sufficient to test implemented derivatives | Could miss rare geometry; also check worst sampled residual points | Diagnostic hypothesis |
| Difference steps 1e-4,1e-5,1e-6 | FP64 truncation/roundoff sensitivity ladder | Crossing ELU branches; compare all steps | Diagnostic hypothesis |
| Invariant tolerance 1e-6 absolute; parameter FD 1e-6+1e-4*|derivative| | Conservative FP64 localization tolerances | Could miss smaller bugs; report actual errors | Engineering screen |
| Radius squared 8 | Existing diagnostic 4D, D=2; exact chi-square tail exp(-4) | Not all such mass lies between modes | Explanatory only |
| 241x241 plot grid | Convenience resolution for a static illustration | Narrow peaks missed; never use plot to calculate probabilities | Illustration |
| Two CPU threads; 1,200 CPU-core-second total allocation | Small suballocation of remaining 8,734.864067 campaign CPU seconds | Stop on budget; at most two attempts, each capped by remaining allocation | Cost cap |

Program: `docs/benchmarks/diagnose_neutra_scientific_root_causes_20261004.py`.
Command: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/diagnose_neutra_scientific_root_causes_20261004.py`.
Output: `docs/plans/artifacts/neutra-scientific-2026-10-04/root-cause-r1/`;
fresh directory per retry. Each attempt records source/input hashes, exact
command, environment, hidden GPUs, seeds, wall/CPU time and this plan. Prior
campaign evidence is preserved. Diagnostic cost is an additional debit from
the same cumulative allocation, never a reset or an added historical grant.

## Skeptical pre-run audit

The important earlier confound was an invalid uncertainty denominator; this
investigation instead uses analytic identities and combined/paired uncertainty.
Exact teacher data isolate student fitting but do not prove infinite-sample
teacher quality. Training/heldout loss and saved learning histories distinguish
obvious overfit from unfinished optimization without claiming an optimum.
The before/after comparison conditions on a single initialization per target;
it can establish what happened to these checkpoints, not a population ranking.
RKL weights and conditional shapes are coupled in a flow: an idealized
fixed-shape reweighting formula will be labeled as such, not asserted to be the
actual neural optimizer trajectory. A finite density grid cannot certify
normalization. Framework comparison checks run before scientific interpretation.
The plan passes this audit with those limits; no material decision requires a
new user direction.

## Local rendering repair

Attempt root-cause-r1 completed all eight checkpoint checks and numerical
summaries in 53.48 seconds, using 60.78 CPU-core seconds. It then failed to
import Matplotlib in the TensorFlow environment. All numerical results and the
executed diagnostic source are preserved. This is a plotting-dependency failure,
not a numerical or scientific veto. The base conda environment already contains
Matplotlib. The repair regenerates only the four illustrative density grids,
uses that existing environment for rendering, and verifies the checkpoint hashes
against the completed numerical evidence. No numerical check or training is
repeated and no package is installed. Review: question, models, checkpoints,
criteria and total allocation remain unchanged.

Repair command:
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/diagnose_neutra_scientific_root_causes_20261004.py --render-only-from docs/plans/artifacts/neutra-scientific-2026-10-04/root-cause-r1 --output docs/plans/artifacts/neutra-scientific-2026-10-04/root-cause-r2`.
