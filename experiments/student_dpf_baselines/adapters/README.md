# Student Baseline Adapters

Adapters normalize student implementations behind a common comparison runner.
They must not be imported by production BayesFilter modules.

Target comparison interface:

```python
run_filter(model, observations, *, seed, num_particles, config) -> BaselineResult
```

The adapter should report unavailable fields explicitly rather than inventing
metrics the student implementation does not expose.
