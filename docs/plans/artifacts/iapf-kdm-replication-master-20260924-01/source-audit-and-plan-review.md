# Source audit and skeptical review of the replication master

Date: 2026-09-24. This is a bounded source audit for two specified methods,
not a comprehensive survey of particle filtering. The master is
[the canonical replication program](../../iapf-kdm-replication-master-program-2026-09-24.md).
No numerical experiment or new training was run to write it.

## Sources inspected and version limits

| Source | Local record / external identity | Inspection and limitation |
| --- | --- | --- |
| Guarniero, Johansen and Lee, The Iterated Auxiliary Particle Filter | .localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf and .txt; DOI 10.1080/01621459.2016.1222291; Warwick record https://wrap.warwick.ac.uk/79711/ | Local copy identifies itself as the accepted manuscript. Algorithms 3–5, fit Equations (15)–(16), and Section 5.2 inspected. Published-version identity has not been established. |
| Younis and Sudderth, Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes | .localresources/papers/younis-sudderth-2023-long-range-tracking.pdf and .txt; https://arxiv.org/abs/2404.08789 | Local text is arXiv v1, 12 April 2024, for the NeurIPS 2023 paper. Sections 4–6 and relevant Appendices B/D inspected. The arXiv page lists v1; no later version was listed at this check. |
| Official proceedings record | https://proceedings.neurips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract.html | NeurIPS 2023 title/authorship confirmed. Proceedings-versus-arXiv technical equivalence has not been checked. |
| Original MDPF implementation | https://github.com/asyounis/mdpf_neurips_2023 ; local .localresources/code/younis-mdpf-neurips-2023 ; commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18 | Commit verified; README and exact base MDPF configuration, model factory, IWSG branch, training includes and trial generator inspected. No full author training executed in this audit. |
| Independent public iAPF implementation | https://github.com/Sempreteamo/iAPF ; pinned a88114395f6c11075fedc653db9480791b43391a in the September 18 comparison | Not verified original-author code; missing declared license and known target differences prevent treating it as the paper oracle. Existing local comparison supplies its audit; source not promoted or redistributed. |

Web checks on 2026-09-24 confirmed the MDPF author repository, arXiv version
listing and conference identity. A publisher-domain iAPF correction query
returned no relevant correction in the inspected results. This is not proof
that no erratum exists. Version-of-record/errata reconciliation remains a
Stage 0 obligation before an exact published-version claim.

Technical claims here come from the inspected local method and experiment
sections and code, not the web abstracts. Citation-count metadata was not
retrieved: selection is by the owner's two methods and original authorship,
not popularity. No citation-completeness claim is made.

## Load-bearing source anchors

Line numbers refer to the local pinned text/code, prior to future source changes.

| Claim | Anchor | Verdict and consequence |
| --- | --- | --- |
| iAPF is an adaptive repeated filtering/backward-fit procedure | iAPF text lines 650–690, Algorithms 3–4 | Supported. A single fitted-twist call is insufficient. |
| Published fit uses density residuals and then a positive floor | iAPF text lines 911–946, Equations (15)–(16) | Supported. QR/score/relative-shape objectives are not interchangeable. |
| The unrestricted fit has a vanishing-amplitude direction | Equation (15): choose covariance sI with s tending to infinity and positive lambda tending to zero on a finite cloud | Local deduction: every residual tends to zero. This proves an escape direction, not the behavior of a particular constrained/local optimizer. |
| Skipped resampling retains accumulated weights | iAPF text lines 951–1020, Algorithm 5 | Supported. Every-step resampling changes the finite program. |
| Initial scope and published counts | iAPF text lines 1022–1097, Section 5.2/Table 1 | Supported for the stated study. Original seeds/data and original optimization settings remain unresolved. |
| IWSG uses a fixed proposal and fixed draws | MDPF text lines 448–470, Equations (14)–(15) | Supported. Pointwise derivative parity does not establish an unbiased score of an unrelated model likelihood. |
| MDPF is a learned full filter | MDPF text lines 535–548 and 624–661, Sections 5–6 | Supported. Component KDE agreement is insufficient. |
| Small full-sequence learning experiment | MDPF text lines 929–986, Appendix B | Supported. Exact executable synthetic source route still needs identification. |
| Bearings protocol and training details | MDPF text lines 624–661, 1610–1648 and 1724–1754; Appendix D and Section 6 | Supported for the inspected details. Remaining architecture/preprocessing and source-paper differences must be resolved before execution. |

## Correct MDPF call chain and launch hazards

All following paths are relative to the pinned author repository's src/.

1. experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/experiment0002_importance/config.yaml:
   lines 7–20 select KDEParticleFilter and differentiable ImportanceSampling;
   later fields select Normal, Normal and Von_Mises kernels.
2. models/model_creator.py:29–30 selects KDEParticleFilter for that type.
3. models/kde_particle_filter/kde_particle_filter.py:675 begins resampling;
   lines 738–748 sample detached locations, evaluate the full KDE density,
   inject exp(log_prob - detached_log_prob), and add the source epsilon.
4. The same class's weight path multiplies inherited resampling factors at
   line 928 and normalizes at line 934; forward begins at line 972 and calls
   weighting near line 1056.
5. The exact trainer-to-forward-to-loss chain, both train and evaluation modes,
   and gradient arrival at each network still need executable wiring/parity
   evidence. **That full call-chain question is not checked by this static audit.**

The similarly named importance_sampling_pf_learned_band/config.yaml instead
selects ImportanceSamplingParticleFilterLearnedBand. The factory maps that
name at lines 57–58 to the discrete comparator class, whose resampling draws
ancestors without continuous kernel noise. Treating it as MDPF would reproduce
the wrong method.

The base MDPF configuration imports configs/generated/our_experiments.yaml,
which was absent at inspection. The generator
scripts/generate_multiple_runs_config_files.py uses NUMBER_OF_TRIALS = 11.
The template configs/our_experiments.yaml has training disabled at lines 17
and 53 and an evaluation switch enabled at line 103. It references pretrained
observation/weight encoders from individual_training. The inspected
configs/common_parameters_training.yaml supplies 800 maximum epochs,
four-step truncated BPTT, N = 25, batch 64, Adam, clipping and validation
scheduling. A direct successful invocation could therefore fail on missing
files or evaluate an existing checkpoint without conducting the intended
training. Stage 0 must materialize and validate the effective experiment.

The prior September 18 comparison already identified density smoothing and
resampling-weight epsilons. Preserve that distinction. A paper-level ideal
IWSG identity and the floating-point author implementation are separate
comparison targets wherever their proposal/density conventions differ.

## Reused evidence and scope

- [September 18 comparison](../../younis-iapf-kdm-reference-comparison-2026-09-18.md)
  and its [result](../younis-iapf-kdm-reference-comparison-20260918-01/result.md):
  component scope, 66 passing KDM checks, explicit nonclaims about learned-model replication.
- [September 20 R paper plan](../../iapf-r-paper-replication-2026-09-20.md):
  learned-fit replication unresolved despite passing exact-twist controls.
- [September 21 fit-resolution plan](../../iapf-r-equation15-resolution-2026-09-21.md):
  original objective versus alternate fit definitions and solver diagnosis.
- [Recovered priority and prior status](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-xla-execution-audit-20260923-01/replication-first-status.md):
  exact session/user anchors and the distinction between two replications and the later integration study.
- [XLA investigation](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-xla-execution-audit-20260923-01/result.md):
  compilation, tracing, timeout and host-scheduling evidence for the old local route, not original-paper replication evidence.

Backward source checking here followed the papers into their explicit
algorithms, experimental baselines and appendices and the author repository's
actual configuration/class chain. It did not audit every cited baseline's
original paper. The local 2024 Learning to Be Smooth paper concerns later
smoothing work and is deferred; no result from it is used to certify the
2023 filter. Broader forward citation searching is deferred because it does
not resolve the immediate question of which original methods must run.
Any later import of a newer method requires its own source check.

The material omission risks are original iAPF implementation/numerical choices,
exact published-version differences, original data/seeds, the complete
synthetic MDPF runner, and the full bearings dependency/training chain.
These remain visible work items rather than assumed facts.

## Skeptical plan audit

| Risk sought | Finding | Resolution in the master |
| --- | --- | --- |
| Wrong baseline/method | Relative-shape or QR iAPF and discrete importance-resampling code can look like the requested methods | Original objectives and exact MDPF configuration named; different variants cannot close replication |
| Proxy promoted to success | Component checks, finite outputs and XLA success were liable to stand in for full replication | Source, mechanics, experiment, scientific result and port statuses separated |
| Missing stopping/budget rule | Full paper effort cannot be inferred from tiny timing runs | Stage 0 capped; full reference/training allocation must be measured and frozen before launch |
| Unfair comparisons | Different objectives, data, initialization, fit effort, training runs and test-based choices | Prespecified matched comparisons; cost and information advantages reported separately |
| Hidden defaults | iAPF floor/optimizer, MDPF pretraining/configuration, epsilon and TF precision | Explicit assumption table and early diagnostics |
| Stale context | Former master begins with a precision repair or an old live supervisor | Authority notices redirect to this master; precision continuation deferred |
| Environment mismatch | Original reference code differs from local TF runtime and GPU sandbox behavior | Separate reference role, exact environment/probes, no unapproved installs |
| Misleading successful command | Missing generated YAML and disabled training could bypass the intended experiment | Effective-config and call-chain checks required before the benchmark |
| False mathematical conclusion | IWSG component identity confused with a general model-likelihood score | Explicit gradient target and proposal assumptions |
| Overstrong statistical claim | Few seeds, dependent time steps, or an arbitrary factor around a published SD | Independent units, uncertainty, margins before runs, and inconclusive status |
| Scope drift | Whole-paper claims despite two selected experiment families | Exact first scope and deferred examples declared |
| Unnecessary governance | More plans could become more permission/review chains | One master/checkpoint, ordinary provenance, one bounded next tranche |

Review verdict: **adequate to establish the work order and begin bounded
source closure; not ready for full numerical replication campaigns**.
This is a self-review, not an independent review. Executable whole-pipeline
conformance and measured full-run budgets remain outstanding. No scientific
completion, method ranking or production-readiness claim is made.

## Governance migration

The current owner direction and current AGENTS.md supersede older procedural
gates and the previous “precision repair first” queue. Historical plans,
results, remaining balances and failure notes remain preserved. The broad
score master and its checkpoint are redirected rather than deleted or
rewritten. The new worktree master is only a pointer to the one planning
authority in the main checkout. No runtime, numerical default, environment
or experiment artifact is modified by this documentation migration.

## Documentation validation

The inspected author-code paths have no working-tree modifications against the
recorded commit. Local links in the new master, checkpoint, review and worktree
pointers were checked. Authority notices preserve all five original document
bodies byte for byte; hashes are recorded in redirect-preservation.json.
Scoped git diff --check passed in both checkouts. The validation record is
[validation.json](validation.json). This is documentation/source inspection
only; no numerical test, training run, runtime edit or default change occurred.
