"""Small CPU diagnostic checks of pilot decisions; no q20 fit claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pytest
import tensorflow as tf

from bayesfilter.testing.q20_neutra_pilot import AffineControl, TeacherGeometry, legacy_affine_chart, summary, teacher_screen
from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig, NeuTraOptimizerConfig


def population(**changes):
    return dict(ess=100.,max_weight=.01,negative_mass=.5,mean=[0.,0.,0.,0.],**changes)


def test_teacher_rejects_population_degeneracy_despite_pooled_coverage():
    rows=[population() for _ in range(8)]
    rows[0]['ess']=1.5
    assert not teacher_screen(rows)['passed']
    rows[0]['ess']=100
    rows[4]['negative_mass']=1.
    assert not teacher_screen(rows)['passed']


def test_teacher_uses_population_replicates_to_detect_partition_bias():
    rows=[population() for _ in range(8)]
    assert teacher_screen(rows)['passed']
    for row in rows[4:]:row['negative_mass']=.8
    result=teacher_screen(rows)
    assert result['weight_screen'] and not result['replication_screen']
    assert result['mass_standard_error']==0.


def test_importance_summary_invariant_to_constant_log_weight_shift():
    x=tf.constant([[0.,0.,-1.,0.],[2.,0.,1.,0.]],tf.float64)
    a=summary(x,tf.math.log(tf.constant([1.,3.],tf.float64)))
    b=summary(x,tf.math.log(tf.constant([1.,3.],tf.float64))+123.)
    assert a['negative_mass']==pytest.approx(.25)
    assert a['ess']==pytest.approx(1.6)
    assert b['mean']==pytest.approx(a['mean'])
    assert b['negative_mass']==pytest.approx(a['negative_mass'])


def test_gaussian_control_has_exact_zero_transformed_score():
    center=tf.constant([1.,2.,3.,4.],tf.float64)
    factor=tf.constant([[1.,0.,0.,0.],[.3,2.,0.,0.],[.2,.1,1.,0.],[0.,.2,0.,3.]],tf.float64)
    cov=factor@tf.transpose(factor)
    flow=AffineControl(center,cov)
    z=tf.random.stateless_normal([4,4],[1,2],dtype=tf.float64)
    x,ld=flow.forward_and_logdet(z)
    score=-tf.transpose(tf.linalg.solve(cov,tf.transpose(x-center)))
    back,ild=flow.inverse_and_forward_logdet(x)
    tf.debugging.assert_near(back,z,atol=1e-12)
    tf.debugging.assert_near(ild,ld,atol=1e-12)
    tf.debugging.assert_near(flow.pullback_score_batch(z,score)+z,tf.zeros_like(z),atol=1e-12)


def test_forward_only_training_does_not_call_expensive_target():
    def forbidden_target(x):
        raise AssertionError('forward objective must not call q20 target')
    cfg=NeuTraTransportConfig.huang_dsf(4,hidden_layers=(4,4),stages=1,mixture_components=2,seed=(2,3))
    flow=NeuTraTransport(cfg)
    trainer=JointNeuTraTrainer(flow,forbidden_target,
        NeuTraOptimizerConfig(4,'standard',.001,.9,.999,1e-8,None,False),
        target_signature='a'*64,teacher_id='tiny_cpu_reference',forward_weight=1.,reverse_weight=0.)
    z=tf.random.stateless_normal([4,4],[3,4],dtype=tf.float64)
    before=[tf.identity(v) for v in flow.trainable_variables]
    result=trainer.train_joint_step(z,z,tf.zeros([4],tf.float64))
    assert bool(result['valid'])
    assert any(bool(tf.reduce_any(a!=b)) for a,b in zip(before,flow.trainable_variables))


def test_legacy_chart_density_includes_mixture_spread_and_jacobian():
    geometry={'representatives':{},'source_curvature':{}}
    for label,sign in (('plus',1.),('minus',-1.)):
        geometry['representatives'][label]={'position':[sign*2.,0.,0.,0.]}
        geometry['source_curvature'][label]={'records':[{'precision':tf.eye(4,dtype=tf.float64).numpy().tolist()}]}
    center,factor,ld=legacy_affine_chart(geometry)
    tf.debugging.assert_near(center,tf.zeros([4],tf.float64),atol=1e-12)
    tf.debugging.assert_near(factor@tf.transpose(factor),tf.linalg.diag(tf.constant([5.,1.,1.,1.],tf.float64)),atol=1e-12)
    assert float(ld)==pytest.approx(.5*__import__('math').log(5.))
    z=tf.constant([[1.,2.,3.,4.]],tf.float64)
    physical=center+z@tf.transpose(factor)
    physical_log_density=-.5*tf.reduce_sum(physical*physical,axis=1)
    stored_chart_log_density=physical_log_density+ld
    tf.debugging.assert_near(stored_chart_log_density-ld,physical_log_density,atol=1e-12)


def test_inverse_teacher_diagnostic_uses_checked_scores_and_handles_padding(tmp_path):
    points=tf.random.stateless_normal([40,4],[17,29],dtype=tf.float64)
    for i in range(4):
        (tmp_path/f'validation-{i}-scores.tensor').write_bytes(tf.io.serialize_tensor(-points[10*i:10*i+10]).numpy())
    class ForbiddenTarget:
        def value_score(self,x):
            raise AssertionError('replayed teacher scores must avoid repeated UKF calls')
    control=AffineControl(tf.zeros([4],tf.float64),tf.eye(4,dtype=tf.float64))
    probe=TeacherGeometry(control,ForbiddenTarget(),(points,tf.zeros([40],tf.float64)),tmp_path)
    result=probe(tmp_path,'gaussian')
    assert result['rows']==40 and result['finite']
    assert result['max_residual']==pytest.approx(0.,abs=1e-12)
    assert result['conditional']['negative']['mass']+result['conditional']['positive']['mass']==pytest.approx(1.)
