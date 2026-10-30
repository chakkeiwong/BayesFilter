# Short-pilot timing correction

The pilot calls sample_neural_force_hmc once for each measured two-transition
chunk. That function computes the initial retained force before the transition
loop. With L=1, two transitions therefore use three force calls. Dividing by
two and extrapolating is a short-chunk average, not a long-chain rate or a
warmup lower bound. No scientific kernel or frozen scope source changed.

The master now emits pilot-timing-audit.json and an explanatory consumer cost
model. Preserved LGSSM artifacts are interpreted in
lgssm-timing-correction-01.json: measured grouped four-chain force40.3684s and
endpoint0.08565s give40.4541s per transition at L=1, with one initial force.
A2000-transition extrapolation is22.4857h under those fixed-position costs.
Adaptive tuning, longer trajectories, and changing positions are unmeasured;
2000 transitions do not establish readiness or convergence.

Validation:144 orchestration tests passed, timing-reporting-tests-01.log.
The actual LG artifacts satisfy the helper's health requirements and its
three-force count. Source anchor: neural_force_hmc.py::sample_neural_force_hmc,
initial _evaluate_force before tf.while_loop. Existing counter regression
checks one initialization plus transitions times leapfrog length.

Decision: correct reporting and continue fresh model scopes. No inference,
long-chain rate, or speed ranking is established by this correction.
