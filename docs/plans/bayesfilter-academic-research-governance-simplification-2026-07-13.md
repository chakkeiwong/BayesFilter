# BayesFilter Academic Research Governance Simplification

Date: 2026-07-13

Status: `ACTIVE_OWNER_DIRECTIVE`

## Decision

BayesFilter will use proportional academic-research governance. The repository
remains strict about scientific validity, reproducibility, compute budgets,
artifact preservation, privacy, and external or irreversible actions. It will
not use production-service launch-security machinery for ordinary trusted local
experiments unless a concrete threat model justifies it.

This decision is binding through `AGENTS.md` and supersedes conflicting
procedural requirements in older plans and runbooks. Historical evidence is
preserved as history; it does not impose its former approval ceremony on future
runs.

## Problem Being Corrected

The HMC semantic-identity lane accumulated hash-bound approval prose, one-use
authority and claim files, custom manifests, descriptor/inode retirement checks,
immutable empty reservations, and repeated reviews of closely related
governance documents. Those mechanisms defended against adversarial file and
authority substitution that was never part of this academic repository's
declared threat model.

The ceremony caused real harm: infrastructure defects consumed approvals,
empty evidence files became immutable blockers, and scientific work stopped
before workers or HMC transitions. This is a governance-design failure, not a
scientific requirement.

## Skeptical Audit

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | The trusted-local academic threat model matches the repository and owner direction. External/irreversible actions remain separately gated. |
| Proxy promoted | Governance simplification does not relax convergence, posterior, uncertainty, source-faithfulness, or promotion criteria. |
| Missing stop | Scientific invalidity, missing diagnostics, budget exhaustion, privacy/external actions, destructive changes, and material contract drift remain stops. |
| Hidden assumption | The workspace is trusted but fallible. Git, versioned outputs, checksums, tests, and run manifests address accidental error. |
| Stale context | Existing Phase 7 failures were inspected: both stopped in launcher/governance infrastructure before scientific evidence was produced. |
| Environment mismatch | Hardware and framework provenance remain required in serious run manifests. |
| Artifact insufficiency | A versioned run directory plus ordinary manifest preserves commands, commit, environment, seeds, outputs, and checks without custom authority schemas. |

Audit verdict: `PASS_FOR_POLICY_MIGRATION_NO_EXPERIMENT_EXECUTION`.

## Controls Retained

- skeptical scientific-plan audit and evidence contract;
- explicit baseline, promotion criterion, vetoes, explanatory diagnostics, and
  nonclaims;
- statistical uncertainty discipline and direct scientific language;
- serious-run manifest with commit, command, environment, seeds, hardware,
  budget, wall time, and output paths;
- versioned append-only run directories and ordinary integrity hashes;
- focused tests and numerical checks;
- external, destructive, privacy, credential, package/environment, public
  claim, and materially expanded compute approvals; and
- GPU/CUDA trusted-context rules.

## Controls Retired By Default

- exact or hash-bound approval wording;
- per-launch authority or permanent claim artifacts;
- inode/descriptor/hard-link security protocols;
- immutable empty output reservations;
- custom cryptographic artifact schemas used only for launch permission;
- reapproval after infrastructure-only repair inside an unchanged campaign;
- mandatory review of every procedural artifact; and
- review nonconvergence as a stop when no scientific or material engineering
  issue remains.

## Execution Model

| Tier | Scope | Default governance |
| --- | --- | --- |
| Routine | Local code, tests, focused diagnostics, short smokes | User task plus normal tool permissions and focused verification |
| Serious campaign | Long HMC/ML/benchmark work or research decisions | One concise plan, evidence contract, total compute/attempt budget, versioned run root, plain-language execute/resume direction |
| External/irreversible | Publication, messages, secrets, destructive actions, broad environment changes, expanded spending/direction | Explicit human approval at the actual boundary |

Infrastructure repair and retry are allowed inside a serious campaign when the
scientific contract and total budget do not change. The result must record each
attempt, failure classification, repair, and remaining budget.

## Current Phase 7 Migration

The prior Phase 7 proposal, terminal manifest, authority code, claim code, and
attempt-1 evidence remain historical and must not be overwritten. They are no
longer active prerequisites for a new run.

The active Phase 7 route is the academic campaign subplan:
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md`.

Its scientific contract remains unchanged. Its campaign budget is eight total
wall-clock hours across at most three launches, including infrastructure
retries. A future plain-language request such as "resume Phase 7" or "execute
the current Phase 7 campaign" is sufficient to start it. This policy-change
request does not itself launch the experiment.

## Shared Policy

The project-independent form of this profile is maintained in
`~/python/claudecodex/policies/global-scientific-coding-agent-policy.md` and is
installed into machine-level Claude/Codex policy files by
`~/python/claudecodex/install_global_agent_policy.py`.

## Nonclaims

This policy does not establish HMC convergence, posterior correctness,
scientific validity, production readiness, GPU readiness, NeuTra readiness, or
superiority. It changes governance proportionality, not the scientific bar.
