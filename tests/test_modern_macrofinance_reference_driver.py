"""Focused reporting/continuation checks; independent reference only."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("reference_driver",
    Path(__file__).resolve().parents[1]/"scripts/diagnostics/run_modern_hmc_macrofinance_reference.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def rung(power, mean, mcse=.001):
    return dict(power=power,mean=[mean],sd=[1.],randomization_mcse=[mcse])


def test_reference_requires_two_refinements_and_keeps_instability():
    assert not driver.reference_envelope([rung(12,0),rung(14,0)])["passed"]
    report=driver.reference_envelope([rung(12,0),rung(14,.1),rung(16,.1)])
    assert not report["passed"]
    assert report["max_normalized_uncertainty"]==pytest.approx(.103)
    assert driver.reference_envelope([rung(12,0),rung(14,.001),rung(16,.002)])["passed"]


def test_reference_uncertainty_and_missing_scale_cannot_pass():
    assert not driver.reference_envelope([rung(12,0),rung(14,0),rung(16,0,.01)])["passed"]
    bad=rung(16,0);bad["sd"]=[0.]
    with pytest.raises(ValueError,match="scale"):
        driver.reference_envelope([rung(12,0),rung(14,0),bad])


def test_repaired_reference_stream_inventory_is_disjoint():
    seeds=driver.reference_seeds(True)
    used=[seeds['pricing']]
    for base in seeds['calibration']: used.extend(range(base,base+8))
    for base in seeds['reference_rung_bases'].values(): used.extend(range(base,base+64))
    assert len(used)==len(set(used))
    assert set(used).isdisjoint(range(202610290,202620298))
