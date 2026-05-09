# Phase M4 result: differentiable resampling and optimal transport chapter plan

## Date

2026-05-09

## Purpose

This note defines the mathematical chapter structure for differentiable
resampling, OT resampling, and learned/amortized transport operators.

## Core conclusion

Differentiable resampling is not merely a technical appendix to particle flow.
It is a separate mathematical problem with its own exactness, bias, and target
questions.  It therefore deserves a dedicated chapter and possibly a second
follow-on chapter or major section for learned/amortized OT.

## Proposed chapter A: differentiable resampling and OT foundations

### Mathematical role

Explain why standard resampling is not pathwise differentiable and develop the
main smooth alternatives with explicit bias/relaxation interpretation.

### Required sections

1. **Resampling as a discontinuous map**
   - categorical ancestor selection;
   - why pathwise gradients fail.

2. **Soft resampling constructions**
   - define the soft-resampling family;
   - derive what moments are preserved and where bias enters.

3. **Resampling as transport / projection**
   - formulate weighted-to-uniform resampling in measure/transport language.

4. **Entropic OT resampling**
   - state the primal problem;
   - derive or carefully present the Sinkhorn scaling form;
   - define the barycentric projection.

5. **Bias versus differentiability analysis**
   - compare soft and OT resampling as relaxed maps;
   - identify the quantity being approximated and how the approximation depends
     on regularization or interpolation parameters.

### Main equations / objects

- discontinuous ancestor-selection map;
- soft-resampling formula;
- OT / EOT primal and Sinkhorn form;
- barycentric map;
- bias statements or approximation decompositions.

### Exact / approximate distinction

- standard multinomial resampling: exact resampling mechanism but not pathwise
  differentiable;
- soft resampling: smooth but biased surrogate;
- OT resampling: differentiable relaxed transport, not identical to exact
  multinomial resampling.

### Implementation implications to record

- differentiability source;
- need for Sinkhorn iterations or equivalent solver;
- sensitivity to regularization and solver budget;
- effect on value and gradient interpretation.

## Proposed chapter B: learned / amortized OT and implementation mathematics

### Mathematical role

Treat learned transport operators as a further approximation layer on top of OT
or differentiable resampling rather than as a free acceleration.

### Required sections

1. **Teacher-student structure**
   - define what map is learned;
   - make explicit that the learned operator approximates a previously defined OT
     or transport target.

2. **Approximation residual and induced target shift**
   - describe how residual error should be interpreted mathematically;
   - distinguish approximation to the resampling map from approximation to the
     posterior itself.

3. **Training-distribution dependence**
   - state why a learned operator may fail under extrapolation in state dimension,
     weight geometry, or regularization range.

4. **Implementation-facing implications**
   - architecture dependence, state-dimension constraints, compile benefits,
     and risks of silently changing the target.

### Main equations / objects

- teacher OT map or barycentric transport target;
- learned operator as approximation map;
- approximation residual metrics;
- possibly target-shift interpretation in posterior terms.

### Exact / approximate distinction

- learned OT is an approximation to an already relaxed OT map unless a stronger
  theorem is supplied;
- therefore it introduces an additional approximation layer beyond OT itself.

### Implementation implications to record

- training-distribution mismatch risk;
- speedup versus exact/teacher OT;
- need for explicit approximation labeling in production-oriented systems.

## Required comparison structure

| Method | Mathematical object | Source of differentiability | Source of bias / approximation | Implementation burden | HMC interpretation |
| --- | --- | --- | --- | --- | --- |
| Standard resampling | exact categorical ancestor map | none pathwise | none in resampling mechanism itself | low | non-pathwise, not direct gradient path |
| Soft resampling | interpolated ancestor/mean map | explicit smooth interpolation | interpolation-induced target change | low to moderate | surrogate or approximate-target path |
| OT / EOT resampling | entropically regularized transport map | Sinkhorn / differentiable transport solver | regularization and finite-solver approximation | moderate to high | approximate or relaxed target path |
| Learned / amortized OT | learned approximation to OT map | neural forward operator | OT relaxation plus learning approximation | high upfront, low inference | additional surrogate layer |

## Load-bearing derivation obligations from this phase

1. formal statement of resampling non-differentiability;
2. soft-resampling moment/bias analysis;
3. transport formulation of equal-weight resampling;
4. entropic OT primal and Sinkhorn scaling structure;
5. barycentric projection formula;
6. mathematical interpretation of learned-operator residuals.

## Audit

This phase clarifies the key reason the earlier DPF draft was too loose: it did
not sufficiently isolate the differentiability-versus-bias problem.  The rebuilt
structure now treats resampling relaxations as their own mathematical topic,
which is necessary for any later HMC interpretation.

## Next phase justified?

Yes.

Phase M5 is justified because once resampling and OT are isolated, the HMC
question can be answered rung by rung without mixing together unrelated sources
of approximation.
