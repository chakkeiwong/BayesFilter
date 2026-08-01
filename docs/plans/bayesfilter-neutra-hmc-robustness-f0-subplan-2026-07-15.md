# NeuTra HMC Robustness Phase F0 Subplan

Date: 2026-07-15  
Status: `EXECUTED_AND_CLOSED`

## Objective And Entry Conditions

Generate a genuinely new T=120 LGSSM observation fixture with frozen simulation
seed `(20260715, 701)`, bind an exact target signature that differs for the
fixture-bound reason, build target-specific geometry, and produce a freshly
tuned plain-HMC comparator under modern diagnostics and 10,000-sample caps.
S1 passed and C0-C2 remain green.

## Evidence Contract

The question is whether the new posterior target and its plain-HMC comparator
are valid enough to support target-specific NeuTra training and comparison.
The original fixture is the identity/difficulty comparator only. The plain-HMC
candidate is promoted only if its target/status/health gates pass, warm-up is
retained and excluded, modern R-hat is `<=1.01`, bulk ESS is `>=1000`, tail ESS
is `>=400`, and truth recovery is within 3 posterior SD for every parameter.

Fixture value/score, geometry residuals, acceptance, runtime, and descriptive
difficulty measures are explanatory or veto diagnostics, not posterior
promotion criteria by themselves. No claim that the new draw is harder is made
unless declared descriptive diagnostics support that label. No sampler
superiority, NeuTra quality, calibration, broad robustness, production, or
default-readiness claim may be made.

## Defaults, Risks, And Pre-Mortem

- Model, prior, parameter order, horizon, and raw truth remain unchanged so the
  fixture isolates observation realization.
- The simulation seed is new and frozen before execution.
- Target identity must bind config, exact observations, contract, parameter
  names, source closure, and backend.
- Target-specific geometry may use the existing reviewed quadratic initializer;
  it is an HMC preconditioner, not NeuTra training geometry.
- Short fixed-budget probes may nominate a kernel only. Shared bounded
  sequential sampling makes the comparator decision.
- The run could pass misleadingly if it reused the original fixture, target
  signature, mass, or comparator; explicit unequal-hash/signature checks veto
  each case.
- A comparator failure may be tuning or geometry failure rather than evidence
  against NeuTra. It is nevertheless a continuation veto for F1 because there
  would be no valid new-fixture reference.

## Required Artifacts And Checks

- immutable F0 config and repeated-identical fixture generation;
- fixture hash, observation hash, target signature, and original-fixture
  inequality ledger;
- CPU-hidden/XLA value-score/status gate;
- target-specific geometry/mass artifact and residual/conditioning checks;
- fresh fixed-kernel nomination and shared sequential comparator run;
- separate warm-up/retained archives, convergence/recovery result, exact
  commands, environment, seeds, wall time, and paths;
- F0 close record and F1 subplan.

## Budget, Stop, And Handoff

CPU-hidden fixture, geometry, tuning, and comparator sampling have an aggregate
four-hour budget and one fresh-directory retry only for localized infrastructure
failure. Warm-up and retained sampling each cap at 10,000 per chain. Stop F0 for
target/signature invalidity, missing diagnostics, corrupted artifacts, geometry
invalidity without a reviewed repair, comparator health/convergence/recovery
failure at cap, or budget exhaustion. Only a passing fixture-bound comparator
hands off to F1.

Skeptical audit verdict: `PASS_AFTER_REVISION`. The original driver is suitable
for deterministic fixture, XLA, and geometry construction, but its legacy
16k/40k sampling controller is not. Comparator sampling must use the new
10k-capped route.
