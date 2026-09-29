"""Independent CPU diagnostic finite differences for full A/Q callbacks."""
import math
import pytest
import tensorflow as tf
from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_filter_marginal_loglik


@pytest.mark.parametrize('d', [3, 10])
def test_every_coordinate_callback_total_finite_difference(d):
    spec = FullLGSSMSpec('full_matrix', d)
    theta = spec.default_theta()
    x = tf.reshape(tf.linspace(tf.constant(-.5,tf.float64), .7, 3*d), [3,d])
    dx, means, dmeans, obs = .07+x*.03, x*.2, x*.01-.02, tf.ones([d],tf.float64)*.3
    eps = tf.constant(1e-6,tf.float64)
    signature = [tf.TensorSpec([spec.parameter_count],tf.float64)]
    @tf.function(input_signature=signature,autograph=False)
    def evaluate(v):
        model,_ = spec.model(theta,v)
        def value(th,points,mu):
            mod,_=spec.model(th)
            return tf.concat([tf.reshape(mod.transition_mean_fn(th,points),[-1]),
                mod.transition_log_density_fn(th,points,mu),mod.observation_log_density_fn(th,points,obs),
                tf.reshape(mod.process_covariance,[-1]),tf.reshape(mod.observation_covariance,[-1]),
                tf.reshape(spec.initial_cloud(th,x)[0],[-1])],0)
        fd=(value(theta+eps*v,x+eps*dx,means+eps*dmeans)-value(theta-eps*v,x-eps*dx,means-eps*dmeans))/(2.*eps)
        tangent=tf.concat([tf.reshape(model.transition_mean_tangent_fn(theta,x,dx),[-1]),
            model.transition_log_density_tangent_fn(theta,x,means,dx,dmeans),
            model.observation_log_density_tangent_fn(theta,x,obs,dx),
            tf.reshape(model.process_covariance_tangent_fn(theta),[-1]),
            tf.reshape(model.observation_covariance_tangent_fn(theta),[-1]),
            tf.reshape(spec.initial_cloud(theta,x,v)[2],[-1])],0)
        return tf.reduce_max(tf.abs(fd-tangent))
    errors=[float(evaluate(tf.one_hot(i,spec.parameter_count,dtype=tf.float64))) for i in range(spec.parameter_count)]
    assert max(errors)<2e-8, (d,max(errors))
    assert len(spec.parameter_names)==spec.parameter_count== (17 if d==3 else 157)


@pytest.mark.parametrize('d', [3,10])
def test_predict_first_kalman_and_score(d):
    spec=FullLGSSMSpec('full_matrix',d)
    theta=spec.default_theta()
    obs=spec.simulate(theta,2,190001,jit_compile=False)
    assert tuple(obs.shape)==(2,d)
    @tf.function(input_signature=[tf.TensorSpec([spec.parameter_count],tf.float64)],autograph=False)
    def oracle(th):
        return spec.reference_value_and_score(th,obs)
    value,score=oracle(theta)
    coordinates=range(spec.parameter_count) if d==3 else [0,1,9,10,99,100,101,109,154,155,156]
    for i in coordinates:
        v=tf.one_hot(i,spec.parameter_count,dtype=tf.float64)*1e-5
        fd=(oracle(theta+v)[0]-oracle(theta-v)[0])/2e-5
        assert abs(float(fd-score[i]))<2e-7, (d,i,float(fd-score[i]))
    p=spec.kalman_parameters(theta)
    mean=tf.linalg.matvec(p['transition_matrix'],p['initial_mean'])
    covariance=p['transition_matrix']@p['initial_covariance']@tf.transpose(p['transition_matrix'])+p['process_covariance']+p['observation_covariance']
    chol=tf.linalg.cholesky(covariance)
    white=tf.linalg.triangular_solve(chol,(obs[0]-mean)[:,None])
    exact=-.5*(tf.reduce_sum(white**2)+2.*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))+tf.constant(d*math.log(2.*math.pi),tf.float64))
    assert abs(float(spec.reference_value_and_score(theta,obs[:1])[0]-exact))<1e-12
    # Scalar-observation identity is also the diagonal specialization oracle.
    diagonal_theta=tf.tensor_scatter_nd_update(theta,[[i*d+j] for i in range(d) for j in range(d) if i!=j],tf.zeros([d*(d-1)],tf.float64))
    lower_indices=[d*d+i*(i+1)//2+j for i in range(d) for j in range(i)]
    diagonal_theta=tf.tensor_scatter_nd_update(diagonal_theta,[[i] for i in lower_indices],tf.zeros([len(lower_indices)],tf.float64))
    physical=spec.kalman_parameters(diagonal_theta)
    assert float(tf.reduce_max(tf.abs(physical['process_covariance']-tf.linalg.diag(tf.linalg.diag_part(physical['process_covariance'])))))==0.
    full=spec.reference_value_and_score(diagonal_theta,obs)[0]
    terms=[]
    for i in range(d):
        pars={k:(v[i:i+1] if v.shape.rank==1 else v[i:i+1,i:i+1]) for k,v in physical.items()}
        terms.append(kalman_filter_marginal_loglik(obs[:,i:i+1],**pars))
    assert abs(float(full-tf.add_n(terms)))<1e-11
