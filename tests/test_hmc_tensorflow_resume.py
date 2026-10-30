"""Resume mechanics and resource boundaries; CPU-only diagnostic tests."""
from dataclasses import replace
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_tensorflow_resume as resume
from bayesfilter.inference import hmc_tensorflow_verified as verified
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference.hmc_sequential_verification import combined_chunks
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config


def test_fresh_and_resumed_sampling_share_the_20000_maximum():
    original = tuning_config("resume-policy-test")
    for resume_manifest in (None, "saved/manifest.json"):
        extended = replace(original, verification_max_results=20000,
                           resume_manifest=resume_manifest)
        assert verified.VerifiedTensorFlowHMCConfig.from_payload(extended.payload()) == extended
        with pytest.raises(ValueError):
            replace(extended, verification_max_results=21000)
        with pytest.raises(ValueError):
            replace(extended, warmup_max_results=20000)


def mechanics(monkeypatch):
    # Small counts exercise control flow only; no inference admission.
    config = SimpleNamespace(initial_chain_positions=((0.,0.),)*4,
        nomination=SimpleNamespace(parameter_dimension=2), energy_error_max=1000.,
        warmup_results=4, warmup_max_results=8, warmup_check_window_results=4,
        verification_results=4, verification_max_results=12, check_every_results=4,
        chunk_results=2, warmup_rhat_max=1.05, fresh_seed=(31,79))
    monkeypatch.setattr(verified, "_affine", lambda n: SimpleNamespace(
        position_to_latent=lambda x:x, latent_to_position=lambda x:x))
    monkeypatch.setattr(verified, "_health", lambda *args:dict(finite=True, moved=True))
    monkeypatch.setattr(verified, "_rhat_graph", lambda count, dim: lambda x:
        dict(maximum=tf.fill([2], tf.constant(1. if count>=12 else 1.02, tf.float64))))
    calls = []
    def run(state, seed, count):
        calls.append((seed.numpy().tolist(), state.numpy().tolist()))
        states = state + tf.cumsum(tf.random.stateless_normal([count,4,2], seed, dtype=tf.float64), axis=0)
        zeros = tf.zeros([count,4], tf.float64)
        flags = tf.zeros([count,4], tf.bool)
        return stage._TensorChunk(states, zeros, flags, flags, zeros, ~flags,
            states[-1], tf.constant(.1, tf.float64), tf.zeros([count,4,11], tf.int32))
    return config, run, calls


def exercise(monkeypatch, tmp_path):
    cfg, run, calls = mechanics(monkeypatch)
    interrupted = tmp_path/"interrupted"
    interrupted.mkdir()
    partial = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=tmp_path, records=[], output_root=interrupted,
        permit_new=lambda phase,count:len(calls)<5, run_new=run)
    assert partial["status"] == "BUDGET_EXHAUSTED"
    assert partial["counts"] == dict(warmup=4, verification=6)
    return cfg, run, calls, interrupted, partial


def test_resumption_matches_uninterrupted_states_seeds_and_diagnostics(tmp_path, monkeypatch):
    cfg, run, calls, source, partial = exercise(monkeypatch, tmp_path)
    first_calls = copy.deepcopy(calls)
    resumed_dir = tmp_path/"resumed"
    resumed_dir.mkdir()
    resumed = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=source, records=partial["chunks"], output_root=resumed_dir, run_new=run)
    assert len(calls) == 8  # Three new chunks; none of the five old chunks reran.
    assert calls[:5] == first_calls
    continued_calls = copy.deepcopy(calls)
    calls.clear()
    whole_dir = tmp_path/"uninterrupted"
    whole_dir.mkdir()
    whole = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=tmp_path, records=[], output_root=whole_dir, run_new=run)
    assert resumed["status"] == whole["status"] == "VERIFIED"
    assert continued_calls == calls
    assert resumed["events"] == whole["events"]
    assert resumed["used_seeds"] == whole["used_seeds"]
    for phase in ("warmup","verification"):
        a,b = combined_chunks(resumed["phases"][phase]), combined_chunks(whole["phases"][phase])
        for name in a._fields:
            tf.debugging.assert_equal(getattr(a,name), getattr(b,name))


@pytest.mark.parametrize("damage", ["seed", "count", "index", "initial", "checksum"])
def test_resume_rejects_corruption_before_new_transition(tmp_path, monkeypatch, damage):
    cfg, run, calls, source, partial = exercise(monkeypatch, tmp_path)
    records = copy.deepcopy(partial["chunks"])
    record = records[1]
    if damage in ("seed","count","index"):
        if damage == "seed":
            record["seed"][0] += 1
        else:
            record[damage] += 1
    elif damage == "initial":
        record["archive"]["initial"] = stage._write_tensor(
            source/"wrong-initial.tfbin", tf.fill([4,2], tf.constant(9.,tf.float64)))
    else:
        row = record["archive"]["initial"]
        row["sha256"] = "0"*64
    out = tmp_path/"bad"
    out.mkdir()
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        resume.replay_then_continue(config=cfg, nomination=None, binding=None,
            source_root=source, records=records, output_root=out, run_new=run)
    assert len(calls) == 5


def test_archive_replay_never_runs_a_transition(tmp_path, monkeypatch):
    cfg, run, calls, source, partial = exercise(monkeypatch, tmp_path)
    def forbidden(*args):
        raise AssertionError("replay tried to simulate old transitions")
    replay = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=source, records=partial["chunks"], run_new=forbidden)
    assert replay["events"] == partial["events"]
    assert replay["counts"] == partial["counts"]
    assert len(calls) == 5


def test_public_tuner_resume_skips_nomination_and_fresh_initialization(monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel
    cfg = replace(tuning_config("resume-dispatch-test"), resume_manifest="saved/manifest.json")
    def forbidden(**kwargs):
        raise AssertionError("resume repeated adaptation or fresh warmup")
    monkeypatch.setattr(stage, "_run_tensorflow_hmc_tuning", forbidden)
    monkeypatch.setattr(verified, "_fresh_sequence", forbidden)
    seen = []
    sentinel = object()
    def continuation(**kwargs):
        seen.append(kwargs)
        return sentinel
    monkeypatch.setattr(resume, "resume_verified_tuning", continuation)
    binding = object()
    assert tune_hmc_kernel(adapter=None, initial_position=(0.,0.), config=cfg,
        output_dir="new-output", runner_binding=binding) is sentinel
    assert seen[0]["config"] is cfg and seen[0]["runner_binding"] is binding


def test_unfinished_warmup_continues_without_restarting(tmp_path, monkeypatch):
    cfg, run, calls = mechanics(monkeypatch)
    interrupted = tmp_path/"warmup-interrupted"
    interrupted.mkdir()
    partial = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=tmp_path, records=[], output_root=interrupted,
        permit_new=lambda phase,count:len(calls)<1, run_new=run)
    assert partial["status"] == "BUDGET_EXHAUSTED"
    assert partial["counts"] == dict(warmup=2, verification=0)
    assert not partial["warmup_ready"]
    resumed_dir = tmp_path/"warmup-resumed"
    resumed_dir.mkdir()
    resumed = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=interrupted, records=partial["chunks"],
        output_root=resumed_dir, run_new=run)
    continued_calls = copy.deepcopy(calls)
    assert len(continued_calls) == 8
    calls.clear()
    whole_dir = tmp_path/"whole"
    whole_dir.mkdir()
    whole = resume.replay_then_continue(config=cfg, nomination=None, binding=None,
        source_root=tmp_path, records=[], output_root=whole_dir, run_new=run)
    assert resumed["status"] == whole["status"] == "VERIFIED"
    assert resumed["events"] == whole["events"]
    assert resumed["used_seeds"] == whole["used_seeds"]
    assert continued_calls == calls
    for phase in ("warmup", "verification"):
        got = combined_chunks(resumed["phases"][phase])
        expected = combined_chunks(whole["phases"][phase])
        for name in got._fields:
            tf.debugging.assert_equal(getattr(got,name), getattr(expected,name))


def test_migration_accepts_only_the_preserved_resume_version(monkeypatch):
    current = {"hmc_tensorflow_verified.py":"same", "hmc_tensorflow_resume.py":"new",
               "numerical_kernel.py":"unchanged"}
    monkeypatch.setattr(verified, "_source_hashes", lambda:current)
    prior = dict(current, **{"hmc_tensorflow_resume.py":resume._PRIOR_RESUME_SHA256})
    migration = resume.validate_source_migration(prior)
    assert migration["kind"] == "reviewed_unfinished_warmup_resume_20261018"
    assert migration["numerical_kernel_changed"] is False
    prior["numerical_kernel.py"] = "changed"
    with pytest.raises(ValueError, match="numerical source drift"):
        resume.validate_source_migration(prior)


def test_migration_accepts_only_the_preserved_orchestration_version(monkeypatch):
    current = {"hmc_tensorflow_verified.py":"new", "hmc_tensorflow_resume.py":"resume",
               "numerical_kernel.py":"unchanged"}
    monkeypatch.setattr(verified, "_source_hashes", lambda:current)
    prior = dict(current)
    prior.pop("hmc_tensorflow_resume.py")
    prior["hmc_tensorflow_verified.py"] = resume._PRIOR_VERIFIED_SHA256
    assert resume.validate_source_migration(prior)["numerical_kernel_changed"] is False
    prior["numerical_kernel.py"] = "changed"
    with pytest.raises(ValueError, match="numerical source drift"):
        resume.validate_source_migration(prior)


@pytest.mark.parametrize("resume_hash,kind", [
    (resume._PRIOR_RESUME_SHA256, "reviewed_healthy_legacy_event_replay_20261018"),
    (resume._PRE_REPLAY_REPAIR_RESUME_SHA256, "reviewed_pre_replay_repair_resume_20261018")])
def test_healthy_event_migration_requires_exact_preserved_source_pair(monkeypatch, resume_hash, kind):
    current = {"hmc_tensorflow_verified.py": "current-verified",
        "hmc_tensorflow_resume.py": "current-resume", "numerical_kernel.py": "unchanged"}
    monkeypatch.setattr(verified, "_source_hashes", lambda: current)
    prior = dict(current, **{"hmc_tensorflow_verified.py": resume._LEGACY_EVENT_VERIFIED_SHA256,
        "hmc_tensorflow_resume.py": resume_hash})
    migration = resume.validate_source_migration(prior)
    assert migration["kind"] == kind
    assert migration["numerical_kernel_changed"] is False
    for key in prior:
        changed = dict(prior, **{key: "unreviewed-change"})
        with pytest.raises(ValueError, match="numerical source drift"):
            resume.validate_source_migration(changed)


@pytest.mark.parametrize("verification,retained", [(2250,17750),(10000,10000),(15000,5000),(19000,1000),(20000,0)])
def test_combined_post_warmup_maximum(verification, retained):
    from bayesfilter.inference.surrogate_hmc_resume_program import retained_capacity
    assert retained_capacity(verification) == retained
    assert verification+retained <= 20000


def test_real_public_tuner_resume_and_retained_handoff(tmp_path, monkeypatch):
    """Actual TensorFlow/XLA Gaussian transitions, with one interrupted segment."""
    import math
    from bayesfilter.inference import tune_hmc_kernel, load_verified_tensorflow_hmc_result
    from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation, TargetPotentialEvaluation
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import bind_components
    from bayesfilter.inference import hmc_sequential_verification as sequential
    variance = math.log(2.)**2
    @tf.function(input_signature=[tf.TensorSpec([None,2],tf.float64)],jit_compile=True)
    def force(x):
        return PositionOnlyForceEvaluation(x/variance,tf.ones(tf.shape(x)[:-1],tf.bool))
    @tf.function(input_signature=[tf.TensorSpec([None,2],tf.float64)],jit_compile=True)
    def endpoint(x):
        valid=tf.ones(tf.shape(x)[:-1],tf.bool)
        return TargetPotentialEvaluation(tf.reduce_sum(x*x,-1)/(2*variance),
            dict(value_valid=valid,force_valid=valid,numerical_failure=~valid,outside_declared_support=~valid))
    scope="Gaussian-resume-integration-mechanics-only"
    binding=bind_components(force,endpoint,scope)
    adapter=SimpleNamespace(target_scope=scope,adapter_signature=lambda:scope+":"+binding.binding_hash)
    cfg=tuning_config(scope)
    cfg=replace(cfg,nomination=replace(cfg.nomination,
        mass_window_results=(16,32),step_adaptation_results=64,verification_results=64),warmup_results=2000)
    count=[0]
    def permit(phase,n):
        if phase=="warmup":
            return True
        count[0]+=n
        return count[0]<=250
    monkeypatch.setattr(sequential,"runtime_budget_guard",lambda:permit)
    parent=tune_hmc_kernel(adapter=adapter,initial_position=(0.,0.),config=cfg,
        parameter_scales=(math.log(2.),)*2,output_dir=tmp_path/"parent",runner_binding=binding)
    original=json.loads(Path(parent.artifact_manifest_path).read_text())
    assert not parent.passed and original["sequence"]["counts"]["verification"]==250
    def no_adaptation(**kwargs):
        raise AssertionError("resume must not repeat nomination")
    monkeypatch.setattr(stage,"_run_tensorflow_hmc_tuning",no_adaptation)
    monkeypatch.setattr(resume,"runtime_budget_guard",lambda:None)
    cfg=replace(cfg,resume_manifest=parent.artifact_manifest_path,verification_max_results=20000)
    continued=tune_hmc_kernel(adapter=adapter,initial_position=(0.,0.),config=cfg,
        output_dir=tmp_path/"resumed",runner_binding=binding)
    assert continued.passed
    loaded=load_verified_tensorflow_hmc_result(continued.artifact_manifest_path,
        adapter=adapter,runner_binding=binding)
    assert loaded.passed
    actual=json.loads(Path(continued.artifact_manifest_path).read_text())
    assert actual["sequence"]["chunks"][:len(original["sequence"]["chunks"])]==original["sequence"]["chunks"]
    assert actual["continuation"]["repeats_tuning"] is False
    assert actual["continuation"]["repeats_warmup"] is False
    assert actual["sequence"]["counts"]["warmup"]==original["sequence"]["counts"]["warmup"]
