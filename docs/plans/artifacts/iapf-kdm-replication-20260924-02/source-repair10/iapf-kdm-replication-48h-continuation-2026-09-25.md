# Authorized 48-hour replication continuation

Owner instruction, 2026-09-25 Hong Kong time: "you have 48 more GPU hours.
Is that enough? continue the execution." This authorizes continuation within
the existing replication direction and supersedes the pending three-attempt
request. No further owner approval is needed for local repairs or planned
experiments within this additional time budget.

## Question, scope and sufficient-budget assessment

Can source-faithful iAPF and MDPF/IWSG replication prerequisites be closed,
and which complete source experiments fit the available budget? The original
scoped completion requirements remain unchanged. Do not call a shortened or
reduced-repeat study a completed replication.

The old ledger has 283.127934 GPU-worker seconds charged against 900 seconds.
Add exactly 172800 GPU-worker seconds: cumulative cap 173700 seconds and
available balance 173416.872066 seconds (48.171353 h). CPU diagnostic cap
remains 1800 seconds, 26.154373 used. Preserve every previous attempt and
snapshot in the same output root. Set a cumulative engineering attempt ceiling
of 60 (48 new launches); attempts also consume the time budget. This ceiling
does not authorize wasteful retries. Maintain the current 300-second child cap
until a specific longer-stage protocol states its timeout and evidence.

The measured upper-epoch arithmetic is 81–85 GPU hours for MDPF NLL full
training alone (11 x 800 epochs) and 165–173 hours for DIS-PF's two stages.
Therefore 48 additional hours cannot guarantee completion of either the full
comparison suite or the two-method replication. Source early stopping may
reduce costs but is not assumed to do so. The budget is sufficient for the
missing cost probes, implementation/source closure, smaller experiments, and
selected complete source trials if their contracts and costs are established.
Do not spend the whole budget on partial long runs before resolving the
scientific prerequisites.

## Execution sequence and provisional allocation

1. Resolve TG/IRG cost gaps using existing source-faithful short reference
   consumers. Run mdpf_tg_cost and mdpf_irg_cost, each at most 300 seconds;
   reserve up to 30 minutes total for cost probes under a documented timeout
   extension only if the original source route needs it.
2. Close source checkpoint-to-metric mapping, Appendix B consumer/objective
   provenance, and the iAPF local-fit/floor specification. Use focused
   implementation and mathematical fixtures. Up to 90 GPU minutes for
   discriminating repairs/conformance smokes, with the unchanged CPU balance.
3. From the measured inventory and source closure, specify complete source
   trials or Appendix B experiments that fit up to 40 GPU hours, including
   pretraining, source schedules and independent repeats. Freeze the particular
   experiment contract before launching it. No outcome-driven truncation,
   checkpoint substitution or statistical superiority claim from few trials.
4. Reserve at least 6 GPU hours for untouched evaluation, required comparators,
   source-to-port checks and localized repairs. Unspent allocation can move
   between phases within the same overall cap if the scientific scope stays
   unchanged and the decision is recorded first.

These allocations are maxima to guide planning, not commands to consume all
hours. Source or numerical invalidity stops its dependent lane. A failed
candidate triggers its planned repair; it does not reject the research direction.

## Initial evidence contract and default audit

TG-MDPF and IRG-MDPF comparators must resolve their pinned original author
configurations at commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
The actual model factory, KDE resampler, differentiable setting, loss,
optimizer, source learning rates, 5000/1000 trajectories, T17/N25, batches
64/256 and sparse labels/BPTT modulo four must match their named source arm.
One pretraining epoch, two full-training epochs and one loader worker are
explicit cost-only overrides. Seeds 924011/924012/924013 are reproducible
diagnostic choices, not final trial seeds. Run sequentially on the 4080 SUPER,
using the existing isolated author-reference environment and fixed library path.

Primary pass: actual consumer wiring, finite changed parameters, intended
frozen-parameter invariance, reloadable checkpoints and preserved timing records.
IRG must invoke the original injected-gradient path; TG must not. Nonfinite
values, mismatched source/inputs or missing diagnostics veto that result.
A shared import/environment failure is a continuation veto for dependent jobs.
Runtime and memory are explanatory. No accuracy, ranking, paper replication
or TensorFlow/XLA admission follows. Existing R/PyTorch/NumPy paths remain
independent references; this is not permission to change the runtime backend.

Skeptical audit: the previous failures occurred before training because the
dispatcher omitted the reference C++ library path. The repair passes the actual
CPU author-factory import and all 18 harness tests. The new limit tests confirm
preserved charges, immutable default limits for other campaigns, and rejection
of unrecorded budget changes. The GPU is available and the source probe
selects the same 4080 family by name. Full-budget sufficiency must not be
inferred from two early epochs. Source checkpoint labels remain untrusted until
traced to actual consumers. These checks justify the two bounded cost launches.

Command, from the execution checkout (escalated GPU access):

~~~bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B scripts/run_replication_master.py run --stage stage0 --output docs/plans/artifacts/iapf-kdm-replication-20260924-02 --evidence-root /home/chakwong/BayesFilter --resume --retry-incomplete --continuation-plan /home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-48h-continuation-2026-09-25.md --repair-note "Owner authorized 48 additional GPU hours; campaign-specific ledger limits; TG/IRG source cost continuation" --jobs mdpf_tg_cost mdpf_irg_cost
~~~

Artifacts: existing -02 root, fresh attempt13 onward and new source snapshots.
Save plan, exact command, pinned source, device/environment, seeds, wall time,
checkpoint hashes and results. Result notes must distinguish hard vetoes,
descriptive costs, supported inference, remaining gaps and the next justified
action. No extension to LEDH, NeuTra/HMC or other studies is authorized here.
