"""Small CPU reference allocations, never production sampling defaults."""
import pytest

from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign


@pytest.fixture
def design():
    def make(engine="mechanics", target="gaussian", route="frozen", control="baseline", **overrides):
        values = dict(design_id="unit-reference", engine=engine,
            scenario=ScenarioSpec(target, route, control), replications=16, draws=64,
            seed=15481, budget_seconds=120, purpose="focused engineering regression",
            numerical_provenance="small CPU reference arrays; no statistical promotion",
            device="cpu_reference")
        values.update(overrides)
        return ValidationDesign(**values)
    return make
