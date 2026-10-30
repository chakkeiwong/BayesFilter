# NeuTra historical implementation notice and canonical architecture policy

Owner directive, 2026-09-25: mark all superseded local NeuTra implementations
as **HISTORICAL — UNFAITHFUL TO THE AUTHOR'S CODE**, establish the repaired IAF
architecture as canonical, and require a substantive explanation for deviations.
The requested delivery includes a Git commit, remote merge, conflict resolution
where needed, and push to remote main.

## Scope and historical status

The canonical architecture is `bayesfilter_neutra_iaf_author_v1`, defined in
[the implementation reference](../reference/neutra-implementation.md).
It is the repaired IAF used in the September 24 study, implemented by
`neutra_transport.py` through the single numerical authority
`neutra_transport_core.py`. The configured implementation was introduced in
commit `8e132893e`; the completed study is recorded in `c130c1027` and the
integrated reference tree is `f3995a06a467f16574f96bbc8a68ccbbc4e30dad`.

All earlier local NeuTra architecture implementations and configurations are
historical and unfaithful to the inspected author implementation. This includes
copied or renamed versions, archived execution trees, old checkpoints, affine
substitutes, ordinary/weighted/legacy IAF constructors, and the preliminary
scalar nonlinear-correction canary. The rule applies even if an old plan,
report, class name, or artifact called them canonical, faithful, production, or
qualified. A legacy configuration still has this status when a compatibility
facade evaluates it through the current shared core. A file's date, use of
TensorFlow/XLA, or successful deserialization cannot establish canonical status.

Preserve the original files, arrays, hashes, observations and reports. They
remain historical regression and failure-analysis evidence; they must not be
presented as results of the canonical architecture, used as its defaults, or
silently resumed as the current training recipe. Any deliberate reuse as a
historical comparator or initialization must be labeled and justified under
the deviation requirements below. Loading or converting a checkpoint does not
upgrade its architecture or evidence status.

This classification concerns fidelity and current eligibility. It does not
assert that every old derivative or numerical observation was wrong, or that
architecture differences caused every failure. The continued legacy map also
improved under revised optimization; the completed study did not isolate the
causal contributions of architecture, initialization, estimator, precision and
training duration.

## Implementation inventory

Paths below are relative to `bayesfilter/inference/`.

| Implementation or configuration | Status and permitted interpretation |
|---|---|
| `neutra_transport.py::NeuTraTransport`, configured with the canonical IAF profile | Current canonical architecture; arbitrary configurations of this class are not automatically canonical |
| `neutra_transport_core.py` | Single numerical authority for the canonical map and explicitly classified compatibility/alternative configurations |
| `neutra_training.py` built-in affine, dense and composed maps | Historical, unfaithful author-code variants; retained compatibility/reference behavior |
| `neutra_training_legacy.py` plain dense IAF | Historical, unfaithful author-code variant |
| `neutra_weighted_training.py::WeightedDenseIAFTransport` and its default stages | Historical, unfaithful author-code variant, including the old q20 two/four-stage bounded-tanh recipe |
| `neutra_artifacts.py` old affine/dense schemas and `legacy_neutra_import.py` | Historical readers/importers; successful load is not canonical admission |
| `neutra_training_mechanisms.py` affine/scalar correction maps | Historical diagnostic mechanisms; not the full author IAF/NAF architecture |
| Configured conditional `naf_dsf` with `author_cmade` | Current, separately identified research alternative; not the canonical IAF and not a relabeling of the old scalar canary |
| Tempered, weighted, scale-aware, model-specific and benchmark consumers | Status follows the actual transport configuration and objective; orchestration cannot turn a historical map into a canonical one |
| All older archived copies, plans, runs and checkpoints | Historical under this notice regardless of their original status language |

Generic HMC controllers, target evaluators, affine composition primitives and
artifact readers are not themselves alternative neural architectures. They may
serve the canonical route when supplied the specified configured IAF. This
notice does not require deleting compatibility code or changing old payloads.

## Required explanation before a deviation

Before implementing or running a different architecture or training mechanism,
write a focused note under `docs/plans` that states:

1. The exact configuration/equation difference from the canonical implementation,
   with the paper section/equation and inspected author-source file/line anchors.
   Identify a source option, a necessary local adaptation, or an invention.
2. The concrete mathematical or engineering problem requiring the change, why
   the canonical route is insufficient, and the expected mechanism. Convenience,
   legacy defaults, or a short smoke passing are not sufficient explanations.
3. The affected support, Jacobian, gradients, initialization, optimization,
   precision and downstream HMC behavior, including what is expected to remain
   equivalent and what is deliberately changed.
4. A bounded comparison with the canonical configuration: independent equation
   and derivative checks where applicable, target-specific calibration,
   heldout/seed policy, scientific criteria, failure/stop conditions, compute
   budget and preserved results. State which measurements are explanatory.
5. A skeptical review and the evidence required to retain, reject or promote
   the deviation. Keep it explicitly labeled until that evidence exists.

Ordinary target-specific calibration of width, cap, initialization moments,
optimizer settings, batch size and training duration belongs in the existing
experiment plan with provenance and checks. It must not silently replace the
canonical architecture or turn q20's settings into universal defaults. An
architectural departure requires the fuller explanation above. Local experiments
within existing authorization need no new procedural approval token; replacing
the owner-designated canonical architecture requires explicit owner direction.

## Plan and skeptical audit recorded before editing

Question: can future agents distinguish the canonical architecture from old
local implementations and know what evidence a departure requires?

Baseline: remote main at `f3995a06a467f16574f96bbc8a68ccbbc4e30dad`, whose
reference already documents source correspondence but retains old constructors
without a prominent historical policy. The shared main checkout contains
unrelated work; use an isolated worktree for integration.

Plan: add the policy to `AGENTS.md` and `CLAUDE.md`; specify the canonical
configuration and its source/local-choice boundary in the implementation
reference; mark the numerical entry points and the completed study's comparison
rows; verify the policy against actual source/configuration and focused existing
checks; commit, merge remote main and push without overwriting unrelated work.

Pass criteria: explicit legacy coverage, an executable canonical configuration,
source anchors, a substantive deviation requirement, consistent cross-links,
unchanged numerical behavior, and verified remote integration. Broken links,
misclassified current IAF/NAF results, changed legacy checkpoint meaning or
unrelated staged changes veto delivery. No new training campaign or claim of
posterior accuracy is authorized or needed by this documentation work.

Skeptical audit: a blanket date cutoff would incorrectly discard the current
September 24 IAF fits; classifying by shared-core use would incorrectly promote
legacy maps. The canonical profile must specify masks, initialization and free
scale bias, not merely the label IAF. Width 16, cap 2 and the path-gradient/TF32
training recipe are recorded local choices, not universal author defaults.
The existing documentation's first usage example selects NAF and should instead
show canonical IAF. Historical observations must remain intact. These issues are
addressed by the scope and implementation-reference changes; audit passes for
this bounded policy migration. Review is local; no additional agent is launched.

## Verification and delivery

Verification completed in `/tmp/BayesFilter-neutra-canonical-20260925`:

- `git diff --check` passed.
- Standard-library AST comparison against the integration baseline confirmed
  that all eight Python changes affect only module docstrings. Numerical code,
  constructors, runtime defaults and checkpoint payload semantics are unchanged.
- All three new Markdown links resolve in the integration worktree. Three
  preexisting raw-result links resolve in the shared checkout's local evidence
  root; those large raw artifacts were already intentionally untracked.
- Seven existing focused checks passed, with twenty deselected, in 6.26 seconds:
  free-bias gradients, author block masks, three legacy convention parity cases,
  exact-Gaussian path gradient and the numerical-fork discovery guard. The
  command was:

  ```text
  CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_neutra_single_authority.py -k 'author_free_bias or hoffman_profile_masks or existing_iaf_facades or exact_gaussian_map_has_zero_path_gradient or frozen_and_training_legacy_numerics_share_authority' --junitxml=/tmp/neutra-canonical-policy-20260925-pytest-retry.xml
  ```

  These are small CPU reference fixtures; GPU devices were intentionally hidden.
  The first collection attempt lacked the untracked local native library. The
  retry used `_symmetric_sylvester_ops.so` from the prior execution checkout at
  the identical Git commit, SHA-256
  `2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
  No environment package was installed and no scientific campaign ran. The
  initial failure is preserved in `/tmp/neutra-canonical-policy-20260925-pytest.xml`.

Terminal local review checked the actual constructor/profile, author mask and
scale source, current study configuration, old loader behavior and the policy
scope. It identified and documented the generic constructor's legacy-mask
default and changed the reference's initial example from NAF to canonical IAF.
The policy leaves numerical choices target-specific and preserves the study's
causal and posterior limitations. Architecture selection is the owner's
directive; the seven mechanics checks do not establish fit or posterior quality.

| Decision | Criterion status | Veto status | Main limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| Adopt canonical IAF policy and mark all superseded recipes historical | Source/configuration boundary and entry-point notices explicit | No new-link, numerical-change or focused-test failure remains | Compatibility APIs still load historical maps; labels are agent policy, not a new runtime rejection mechanism | Commit, merge remote main and push; retain policy in shared documents | Automatic migration of every consumer, causal attribution or posterior accuracy |

Git integration uses the isolated branch `codex/neutra-canonical-20260925`.
Only these policy documents and module notices belong in its commit. The
shared checkout receives the targeted policy-document edits while its unrelated
changes, branch position and live numerical sources are preserved. The delivered
commit and any merge are recorded in Git history; remote-head verification is
reported at delivery.
