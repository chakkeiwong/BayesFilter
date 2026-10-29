"""CPU mechanics and design-boundary tests for the scientific campaign."""
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_campaign import (
    CALIBRATION_TARGETS,
    FINAL_TARGETS,
    FIT_SEEDS,
    METHODS,
    profile_candidates,
    target_catalog,
    teacher_screen,
    evaluate_student,
    flow_config,
)
from bayesfilter.inference.neutra_transport import NeuTraTransport


def test_catalog_has_disjoint_calibration_and_final_random_targets():
    catalog = target_catalog()
    assert set(CALIBRATION_TARGETS).isdisjoint(FINAL_TARGETS)
    assert set(catalog) == set(CALIBRATION_TARGETS) | set(FINAL_TARGETS)
    assert len(FIT_SEEDS) == 3
    assert all(len(catalog[name]["centers"]) in (2, 3) for name in catalog if name.startswith("random"))
    assert catalog["fixed_unwarped"]["warp_curvature"] == 0.0
    assert catalog["fixed_warped"]["warp_curvature"] != 0.0


def test_random_design_is_reproducible_and_bounds_are_explicit():
    catalog = target_catalog()
    assert catalog["random_two_1103"] == target_catalog()["random_two_1103"]
    for name in FINAL_TARGETS:
        spec = catalog[name]
        if not name.startswith("random"):
            continue
        centers = spec["centers"]
        distances = [
            ((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2) ** 0.5
            for i in range(len(centers)) for j in range(i)
        ]
        assert all(6.0 <= distance <= 10.0 for distance in distances)
        assert all(0.5 <= variance <= 2.0 for variance in spec["variances"])
        assert all(weight >= 0.1 for weight in spec["weights"])
        assert abs(sum(spec["weights"]) - 1.0) < 1e-12


def test_every_method_has_two_finite_calibration_candidates():
    assert set(METHODS) == {"fab", "gabrie", "ais", "smc", "aft", "craft"}
    for method in METHODS:
        candidates = profile_candidates(method)
        assert len(candidates) == 2
        assert candidates[0]["profile_id"] != candidates[1]["profile_id"]
        assert all(candidate["step_size"] > 0 for candidate in candidates)


def test_exact_teacher_passes_its_own_teacher_screen():
    evaluator = ExactTargetEvaluator(target_catalog()["random_three_2103"], jit_compile=False)
    reference = evaluator.sample(4096, tf.constant([711, 1], tf.int32))
    banks = [(evaluator.sample(1024, tf.constant([711, 2+r], tf.int32)), tf.zeros([1024], tf.float64)) for r in range(4)]
    screen = teacher_screen(evaluator, banks, reference)
    assert screen["finite"]
    assert screen["ess_fraction"] == 1.0
    assert screen["passed"]


def test_heldout_evaluation_accepts_reference_and_probe_banks_with_different_sizes():
    evaluator = ExactTargetEvaluator(target_catalog()["fixed_unwarped"], jit_compile=False)
    flow = NeuTraTransport(flow_config(2, 73))
    report = evaluate_student(
        evaluator,
        flow,
        evaluator.sample(64, tf.constant([733, 1], tf.int32)),
        seed=733, jit_compile=False,
    )
    assert report["finite"]
    assert len(report["q_summary"]) == len(report["reference_summary"])


def test_scientific_wrapper_rejects_extra_arguments():
    wrapper = Path(__file__).resolve().parents[1] / "scripts/run_neutra_scientific_campaign.sh"
    process = subprocess.run(["bash", str(wrapper), "run", "extra"], capture_output=True, text=True, timeout=5)
    assert process.returncode == 2


def test_population_uncertainty_does_not_treat_duplicate_particles_as_independent():
    evaluator = ExactTargetEvaluator(target_catalog()['fixed_warped'], jit_compile=False)
    banks = [(evaluator.sample(64,tf.constant([373,r])),tf.zeros([64],tf.float64)) for r in range(4)]
    reference = evaluator.sample(1024,tf.constant([91,3]))
    original = teacher_screen(evaluator,banks,reference)
    duplicated = teacher_screen(evaluator,[(tf.repeat(x,10,axis=0),tf.repeat(w,10,axis=0)) for x,w in banks],reference)
    tf.debugging.assert_near(original['teacher_mean_variance'],duplicated['teacher_mean_variance'],atol=1e-6)
    assert original['replications'] == duplicated['replications'] == 4
    assert any(v > 0 for v in original['teacher_mean_variance'])


def test_map_uncertainty_includes_both_independent_populations():
    evaluator = ExactTargetEvaluator({'kind':'gaussian','mean':[0.,0.],'covariance':[[1.,0.],[0.,1.]]},jit_compile=False)
    flow = NeuTraTransport(flow_config(2,17))
    report = evaluate_student(evaluator,flow,evaluator.sample(128,tf.constant([713,1])),seed=712,
                              sample_rows=32,jit_compile=False)
    for se,a,b in zip(report['combined_standard_error'],report['q_mean_variance'],report['reference_mean_variance']):
        assert abs(se*se-a-b)<1e-12
    assert report['map_rows']==32 and report['reference_rows']==128


def test_zero_particle_weights_are_valid():
    evaluator = ExactTargetEvaluator(target_catalog()['fixed_unwarped'],jit_compile=False)
    banks = [(evaluator.sample(64,tf.constant([33,r])),tf.concat((tf.zeros([63],tf.float64),tf.constant([-float('inf')],tf.float64)),0)) for r in range(4)]
    report = teacher_screen(evaluator,banks,evaluator.sample(128,tf.constant([5,1])))
    assert report['finite']


def test_tiny_forward_reverse_pipeline_reloads_checkpoints_and_produces_1000_point_probe(tmp_path):
    # Explicit tiny CPU/non-XLA debugging exception; no map-quality conclusion.
    from bayesfilter.testing.neutra_scientific_campaign import run_trial
    profile = {'profile_id':'tiny_reference','width':4,'updates':2,'learning_rate':1e-3,
               'rkl_updates':2,'rkl_learning_rate':3e-4}
    result = run_trial('exact_teacher','gaussian',{'kind':'gaussian','mean':[0.,0.],
                       'covariance':[[1.,0.],[0.,1.]]},profile,123,tmp_path,
                       role='tiny_cpu_reference',exact_teacher=True,jit_compile=False)
    assert result['teacher_admitted']
    assert result['status'] in ('passed','map_failed')
    assert result['forward_training']['updates']==2 and result['rkl_training']['updates']==2
    for label in ('forward','student'):
        assert (tmp_path/(label+'-checkpoint.json')).is_file()
        report=json.loads((tmp_path/(label+'-post-training-1000.json')).read_text())
        assert report['complete'] and report['valid_rows']==1000
    assert result['final']['checkpoint_reloaded'] and result['scientific_promotion'] is False


def test_training_rungs_preserve_stateless_stream_and_optimizer_state():
    from bayesfilter.testing.neutra_scientific_campaign import _train_block
    from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraConfig, WeightedForwardKLNeuTraTrainer
    def build():
        flow=NeuTraTransport(flow_config(2,19,width=4))
        return WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(dimension=2,hidden_layers=(4,4),stages=3,
            learning_rate=1e-3,gradient_clip_norm=1000.,jit_compile=False),transport=flow)
    a,b=build(),build()
    pool=tf.random.stateless_normal([16,2],[33,19],dtype=tf.float64);lw=tf.zeros([16],tf.float64)
    _train_block(a,pool,lw,5,17,batch=8,jit_compile=False)
    _train_block(b,pool,lw,2,17,batch=8,jit_compile=False)
    result=_train_block(b,pool,lw,3,17,batch=8,jit_compile=False)
    assert result['next_rng_counter']==5 and result['traces']==1
    for x,y in zip((*a.variables,*a.optimizer.variables),(*b.variables,*b.optimizer.variables)):
        tf.debugging.assert_equal(x,y)
