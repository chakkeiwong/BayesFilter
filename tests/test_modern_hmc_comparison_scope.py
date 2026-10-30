"""Scope/random-stream regressions for frozen comparison preparation, not inference."""
import itertools
import json
from pathlib import Path

import pytest

from scripts.prepare_modern_hmc_priced_comparison import (
    METHODS, TARGETS, build_designs, comparison_inputs, digest, scoped_seed,
)
from scripts.price_modern_hmc_frozen_design import pricing_seed, worker_command

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def sir_inputs(tmp_path):
    from bayesfilter.inference.hmc_transition_spec import FrozenTransitionSpec, HMCTransitionProfile

    geometry = tmp_path / "full-model-geometry-02-sir/nl-sir"
    write(geometry / "geometry.json", dict(center=[0., 0., 0.]))
    scope = dict(data_sha256="fixture-data", geometry_sha256=digest(geometry / "geometry.json"),
                 horizon=20, parameter_dim=3)
    write(geometry / "design.json", dict(target="NL-SIR", metadata=scope))
    write(tmp_path / "generic-adaptation/preconfirmation-01/reference-consumer-05/NL-SIR.json",
          dict(target="NL-SIR", status="compatible"))
    snaper = tmp_path / "method-proposals-02/nl-sir-snaper/discarded-pilot.json"
    mams = tmp_path / "generic-adaptation/full-model-pricing-02/NL-SIR/pilot/pilot-result.json"
    write(snaper.parent / "design.json", dict(target="NL-SIR", method="snaper", metadata=scope))
    write(mams.parent.parent / "design.json", dict(target="NL-SIR", binding_hash="mams-binding", **scope))
    for path, family in ((snaper, "hmc"), (mams, "mams")):
        profile = HMCTransitionProfile(family, "uniform_duration")
        candidates = [FrozenTransitionSpec(profile=profile, max_steps=length, epsilon=epsilon).payload()
                      for length in ((1, 2, 3) if family == "hmc" else (1,))
                      for epsilon in (.75, .9)]
        pilot = dict(candidates=candidates, config={}, omitted_candidates=[],
                     tuning_artifact_authority=False, fresh_measurement_and_verification_required=True)
        if family == "hmc":
            pilot["adaptation_draws_discarded"] = 1001  # Saved SNAPER schema uses a count.
        else:
            pilot.update(adaptation_and_pilot_draws_discarded=True, binding_hash="mams-binding")
        write(path, pilot)
    return tmp_path, geometry, snaper, mams


def test_target_specific_saved_pilots_and_all_input_hashes(sir_inputs):
    n5, geometry, snaper, mams = sir_inputs
    actual_geometry, _, pilots, paths, hashes = comparison_inputs(n5, target="NL-SIR")
    assert actual_geometry == geometry
    assert len(pilots["snaper"]["candidates"]) == 6
    assert paths == dict(snaper=snaper, mams=mams)
    assert len(hashes) == 7
    assert all(digest(path) == sha for path, sha in hashes.items())


@pytest.mark.parametrize("mutation", ["cross_target", "data", "geometry", "horizon", "dimension",
    "method", "reference", "binding", "not_discarded", "bool_count", "authoritative", "skip_fresh"])
def test_wrong_scope_or_prior_authority_cannot_propose_roots(sir_inputs, mutation):
    n5, _, snaper, mams = sir_inputs
    path = snaper.parent / "design.json"
    data = json.loads(path.read_text())
    if mutation == "cross_target": data["target"] = "NL-PP-UKF"
    elif mutation == "data": data["metadata"]["data_sha256"] = "other-data"
    elif mutation == "geometry": data["metadata"]["geometry_sha256"] = "other-geometry"
    elif mutation == "horizon": data["metadata"]["horizon"] = 10
    elif mutation == "dimension": data["metadata"]["parameter_dim"] = 6
    elif mutation == "method": data["method"] = "randomized_hmc"
    else:
        path = (n5 / "generic-adaptation/preconfirmation-01/reference-consumer-05/NL-SIR.json"
                if mutation == "reference" else mams if mutation == "binding" else snaper)
        data = json.loads(path.read_text())
        if mutation == "reference": data["status"] = "incompatible"
        elif mutation == "binding": data["binding_hash"] = "another-target-binding"
        elif mutation == "not_discarded": data["adaptation_draws_discarded"] = 0
        elif mutation == "bool_count": data["adaptation_draws_discarded"] = True
        elif mutation == "authoritative": data["tuning_artifact_authority"] = True
        elif mutation == "skip_fresh": data["fresh_measurement_and_verification_required"] = False
    write(path, data)
    with pytest.raises(ValueError):
        comparison_inputs(n5, target="NL-SIR")


def test_full_design_preserves_all_roots_repairs_and_separate_posterior(sir_inputs):
    n5, _, _, _ = sir_inputs
    designs, preflights = build_designs(n5, ROOT, target="NL-SIR", seed=72410,
                                      seed_namespace=20261017, plan="scope-test")
    assert set(designs) == set(METHODS) == set(preflights)
    assert [sum(len(e) for _, e in d["search"]["epsilon_by_l"]) for d in designs.values()] == [8, 8, 6, 2]
    for d in designs.values():
        roots = sum(len(e) for _, e in d["search"]["epsilon_by_l"])
        assert d["search"]["max_candidates"] == d["execution"]["acceptance_policy"]["max_candidates"] == 2*roots
        assert d["execution"]["seed"] != d["posterior"]["seed"]
        assert d["proposal_metadata"]["adaptation_replication"] is False
        assert d["posterior"]["sequential"]["minimum_chain_count"] == 4
        assert d["seed"] == 72410


def test_new_scope_needs_namespace_and_wrong_transition_cannot_be_relabelled(sir_inputs):
    n5, _, snaper, _ = sir_inputs
    with pytest.raises(ValueError, match="explicit random namespace"):
        build_designs(n5, ROOT, target="NL-SIR")
    pilot = json.loads(snaper.read_text())
    from bayesfilter.inference.hmc_transition_spec import FrozenTransitionSpec, HMCTransitionProfile
    pilot["candidates"][0] = FrozenTransitionSpec(
        profile=HMCTransitionProfile("mams", "uniform_duration"), max_steps=1, epsilon=.75).payload()
    write(snaper, pilot)
    with pytest.raises(ValueError, match="transition profile"):
        build_designs(n5, ROOT, target="NL-SIR", seed_namespace=20261017)


def test_scoped_streams_are_order_independent_disjoint_and_bounded():
    scopes = list(itertools.product(TARGETS, METHODS, (71410, 72410), ("tuning", "posterior")))
    def stream(scope):
        target, method, seed, stage = scope
        return scoped_seed(target=target, method=method, seed=seed, namespace=20261017, stage=stage)
    forward = {scope: stream(scope) for scope in scopes}
    assert forward == {scope: stream(scope) for scope in reversed(scopes)}
    assert len(set(forward.values())) == len(scopes)
    assert all(0 <= a < 2**31 and 0 <= b < 2**30 for a, b in forward.values())
    other = scoped_seed(target="NL-SIR", method="mams", seed=72410, namespace=20261018, stage="tuning")
    assert other not in forward.values()
    with pytest.raises(ValueError):
        scoped_seed(target="NL-SIR", method="mams", seed=True, namespace=20261017, stage="tuning")


def test_pricing_namespace_preserves_old_streams_and_reaches_worker(tmp_path):
    from argparse import Namespace
    assert pricing_seed(20261012, 2, 3, 1) == (20261012, 820007)
    seeds = {pricing_seed(20261018, i, root, repeat)
             for i, count in enumerate((8, 8, 6, 2)) for root in range(count) for repeat in (0, 1)}
    assert len(seeds) == 48
    args = Namespace(source_root=tmp_path, design=[tmp_path / "design.json"], output=tmp_path / "prices",
                     plan="scope-test", max_wall_seconds=7200., seed_namespace=20261018)
    command = worker_command(args)
    assert command[command.index("--seed-namespace") + 1] == "20261018"
    assert command[-1] == "--worker"
    for bad in (-1, 2**31, True):
        with pytest.raises(ValueError):
            pricing_seed(bad, 0, 0, 0)
