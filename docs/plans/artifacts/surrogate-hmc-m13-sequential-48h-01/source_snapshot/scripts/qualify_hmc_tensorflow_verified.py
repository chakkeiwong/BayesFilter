#!/usr/bin/env python3
"""Bounded GPU qualification of public TF tuning on an exact Gaussian control."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.inference import (
        FrozenPositionOnlyForce, FrozenTargetPotential,
        DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
        bind_neural_force_hmc_tuning_runner, TensorFlowHMCKernelTuningConfig,
        FourChainMeanBandAcceptancePolicy, VerifiedTensorFlowHMCConfig,
        tune_hmc_kernel, build_retained_bound_hmc_archive_runner_from_tuning_result,
        BoundRetainedHMCArchiveConfig, load_verified_tensorflow_hmc_result)
    from bayesfilter.inference.hmc_tensorflow_verified import _source_hashes
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig, neural_force_proposal, TargetPotentialEvaluation
    from bayesfilter.inference.hmc_posterior_diagnostics import posterior_mean_diagnostics, rank_normalized_split_rhat

    report = dict(memory_policy=memory, tensorflow=tf.__version__, source_hashes=_source_hashes(),
                  arms={}, jit_compile=True, purpose="public tuner qualification; no model inference claim")
    for arm, rotation in (("exact", 0.), ("imperfect_rotational", .15)):
        scope = "verified-gaussian-" + arm
        @tf.function(input_signature=[tf.TensorSpec([4,2],tf.float64)],autograph=False,jit_compile=True)
        def force_fn(q):
            return q + tf.constant(rotation,tf.float64)*tf.stack([-q[:,1],q[:,0]],axis=-1)
        force = FrozenPositionOnlyForce(force_fn, identity=scope+"-force-v1",
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
        # Deliberately emit a deterministic diagnostic flag. It does not modify
        # the Gaussian density and must survive the public archive call chain.
        def potential(q):
            return TargetPotentialEvaluation(.5*tf.reduce_sum(q*q,axis=-1),
                {"coordinate_boundary_rounding":tf.ones(tf.shape(q)[0],tf.bool)})
        endpoint = FrozenTargetPotential(potential,identity="standard-normal-2d-flagged-v1")
        binding = bind_neural_force_hmc_tuning_runner(force=force,target=endpoint,target_scope=scope)
        class Adapter:
            target_scope = scope
            def adapter_signature(self):
                return scope + "-adapter-v1"
        adapter = Adapter()
        nominee = TensorFlowHMCKernelTuningConfig(
            parameter_dimension=2,evidence_role="diagnostic_candidate_screen",mass_window_results=(64,128),
            step_adaptation_results=128,verification_results=128,max_leapfrog_steps=4,initial_step_size=.2,
            budget_provenance="bounded Gaussian control qualification",
            initial_step_size_provenance="unit target-scale hypothesis; dual averaging follows",
            geometry_provenance="unit scale warm start; full windowed covariance adapts",
            target_scope=scope,acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
            target_accept_prob=.7,verification_repair_rounds=0,step_repair_factor=2.,mass_shrinkage=.1,
            covariance_jitter=1e-9,eigenvalue_floor=1e-9,max_condition_number=1e8,seed=(20260919,300),use_xla=True)
        cfg = VerifiedTensorFlowHMCConfig(nominee,((-2.,-1.),(-1.,2.),(1.,-2.),(2.,1.)),
                    "four separated unit-scale Gaussian starts", 2000, 2000,(20260919,400),1000.)
        # Deterministic graph/XLA equivalence uses the identical state/momentum,
        # since TensorFlow's graph and XLA random streams need not coincide.
        proposal_cfg = NeuralForceHMCConfig(step_size=.1,num_leapfrog_steps=4,inverse_mass_diagonal=(1.,1.))
        def proposal(q,p):
            result = neural_force_proposal(q,p,force,proposal_cfg)
            return result.position, result.momentum
        signature = [tf.TensorSpec([4,2],tf.float64)]*2
        graph = tf.function(proposal,input_signature=signature,autograph=False,jit_compile=False)
        compiled = tf.function(proposal,input_signature=signature,autograph=False,jit_compile=True)
        q=tf.constant(cfg.initial_chain_positions,tf.float64)
        p=tf.constant([[.3,-.2],[.2,.4],[-.4,.3],[-.2,-.1]],tf.float64)
        ordinary=graph(q,p); xla=compiled(q,p)
        parity=max(float(tf.reduce_max(tf.abs(a-b))) for a,b in zip(ordinary,xla))
        assert parity < 1e-12, parity
        started=time.monotonic()
        result=tune_hmc_kernel(adapter=adapter,initial_position=tf.zeros([2],tf.float64),
            parameter_scales=tf.ones([2],tf.float64),config=cfg,output_dir=args.output/arm/"tuning",runner_binding=binding)
        tuning_seconds=time.monotonic()-started
        report["arms"][arm]=dict(passed=result.passed,checks=result.checks,parity_max=parity,
                                 tuning_seconds=tuning_seconds,manifest=result.artifact_manifest_path)
        (args.output/"report.json").write_text(json.dumps(report,indent=2,default=str))
        assert result.passed, result.checks
        reloaded=load_verified_tensorflow_hmc_result(result.artifact_manifest_path,adapter=adapter,runner_binding=binding)
        runner=build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=reloaded,runner_binding=binding)
        retained=runner.run(BoundRetainedHMCArchiveConfig(2000,(20260919,500),args.output/arm/"retained",
                                                        "bounded exact-Gaussian implementation oracle"))
        assert retained["health_passed"]
        manifest=json.loads(Path(retained["archive_manifest_path"]).read_text())
        samples=tf.io.parse_tensor(tf.io.read_file(str(args.output/arm/"retained"/manifest["raw_samples"]["path"])),tf.float64)
        status=tf.io.parse_tensor(tf.io.read_file(str(args.output/arm/"retained"/manifest["chunk"]["endpoint_status"]["path"])),tf.int32)
        flag_index=manifest["endpoint_status_fields"].index("coordinate_boundary_rounding")
        assert bool(tf.reduce_all(status[...,flag_index]==1)), "endpoint flag lost in public archive"
        report["arms"][arm]["endpoint_flag_archive_wiring"]="PASS"
        diagnostics=posterior_mean_diagnostics(tf.transpose(samples,[1,0,2]))
        mean=tf.reduce_mean(samples,axis=[0,1])
        variance=tf.math.reduce_variance(samples,axis=[0,1])
        # Predeclared broad moment veto for qualification only, not superiority.
        assert bool(tf.reduce_all(tf.abs(mean)<.15)), mean
        assert bool(tf.reduce_all(tf.abs(variance-1.)<.2)), variance
        report["arms"][arm].update(mean=mean.numpy().tolist(),variance=variance.numpy().tolist(),
            rhat={k:v.numpy().tolist() for k,v in rank_normalized_split_rhat(tf.transpose(samples,[1,0,2])).items()},
            mean_diagnostics={k:v.numpy().tolist() for k,v in diagnostics.items()},
            selected_step=float(result.nomination.step_size),selected_L=int(result.nomination.num_leapfrog_steps))
        # Continuing a retained archive must consume a new random stream.
        continuation=runner.run(BoundRetainedHMCArchiveConfig(100,(20260919,501),args.output/arm/"continuation",
                              "continuation wiring check",retained["archive_manifest_path"]))
        assert continuation["health_passed"]
        try:
            runner.run(BoundRetainedHMCArchiveConfig(100,(20260919,500),args.output/arm/"bad-seed",
                        "negative reuse check",retained["archive_manifest_path"]))
        except ValueError as exc:
            assert "already used" in str(exc)
        else:
            raise AssertionError("reused retained seed admitted")
        # A changed endpoint identity must not consume this artifact.
        other=bind_neural_force_hmc_tuning_runner(force=force,target=FrozenTargetPotential(
              endpoint.function,identity="changed-target"),target_scope=scope)
        try:
            load_verified_tensorflow_hmc_result(result.artifact_manifest_path,adapter=adapter,runner_binding=other)
        except ValueError as exc:
            assert "binding mismatch" in str(exc)
        else:
            raise AssertionError("changed target admitted")
        # Alter copies of metadata in place only for these negative checks,
        # restoring exact bytes before any subsequent use.
        tuning_path=Path(result.artifact_manifest_path)
        saved=tuning_path.read_text()
        for field, replacement, message in (
                ("source_hashes",{},"source drift"),
                ("rhat_threshold",1.1,"diagnostic policy"),
                ("checks",{},"diagnostic mismatch")):
            altered=json.loads(saved); altered[field]=replacement
            tuning_path.write_text(json.dumps(altered))
            try:
                load_verified_tensorflow_hmc_result(tuning_path,adapter=adapter,runner_binding=binding)
            except ValueError as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError("changed verification metadata admitted")
            finally:
                tuning_path.write_text(saved)
        altered=json.loads(saved)
        altered["verification"]["states"]["sha256"]="0"*64
        tuning_path.write_text(json.dumps(altered))
        try:
            load_verified_tensorflow_hmc_result(tuning_path,adapter=adapter,runner_binding=binding)
        except ValueError as exc:
            assert "hash" in str(exc) or "checksum" in str(exc)
        else:
            raise AssertionError("corrupted verification tensor admitted")
        finally:
            tuning_path.write_text(saved)
        report["arms"][arm]["artifact_and_continuation_checks"]=True
        report["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
        (args.output/"report.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({"status":"PASS", "arms":list(report["arms"])}))


if __name__=="__main__":
    main()
