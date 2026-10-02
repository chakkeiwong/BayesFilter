"""Full A/full SPD Q LGSSM for explicit SQMC reference comparisons.

The shared canonical executor supplies the analytical recursive score.
Coordinates: A entries, lower-Cholesky Q entries (log diagonal), log R
variance, initial mean scale. Prediction precedes every observation.
"""
from __future__ import annotations
import math
import tensorflow as tf
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel


class FullLGSSMSpec(LGSSMSpec):
    def __post_init__(self):
        if self.family != 'full_matrix' or self.dimension < 2:
            raise ValueError('full_matrix requires dimension >= 2')

    @property
    def parameter_count(self):
        d = self.dimension
        return d*d + d*(d+1)//2 + 2

    @property
    def parameter_names(self):
        d = self.dimension
        return ([f'A[{i+1},{j+1}]' for i in range(d) for j in range(d)] +
                [f'{"log_LQ" if i == j else "LQ"}[{i+1},{j+1}]'
                 for i in range(d) for j in range(i+1)] + ['log_R_variance', 'initial_mean_scale'])

    def default_theta(self, dtype=tf.float64):
        d = self.dimension
        a = [(.65-.20*i/(d-1)) if i == j else .12*(-1)**(i+j)/(d-1)
             for i in range(d) for j in range(d)]
        lower = [math.log(math.sqrt(.16+.08*i/(d-1))) if i == j
                 else .08*(-1)**(i+j)/math.sqrt(d-1)
                 for i in range(d) for j in range(i+1)]
        return tf.constant(a+lower+[math.log(.12), .04], dtype)

    def parts(self, theta, direction=None):
        theta = tf.ensure_shape(tf.convert_to_tensor(theta), [self.parameter_count])
        dtype, d = theta.dtype, self.dimension
        v = tf.zeros_like(theta) if direction is None else tf.cast(direction, dtype)
        a, da = tf.reshape(theta[:d*d], [d,d]), tf.reshape(v[:d*d], [d,d])
        indices = [(i,j) for i in range(d) for j in range(i+1)]
        diagonal = tf.constant([i == j for i,j in indices])
        entries, dentries = theta[d*d:-2], v[d*d:-2]
        entries = tf.where(diagonal, tf.exp(entries), entries)
        dentries = tf.where(diagonal, entries*dentries, dentries)
        lower = tf.scatter_nd(indices, entries, [d,d])
        dlower = tf.scatter_nd(indices, dentries, [d,d])
        q = tf.linalg.matmul(lower, lower, transpose_b=True)
        half = tf.linalg.matmul(dlower, lower, transpose_b=True)
        dq = half+tf.transpose(half)
        r = tf.exp(theta[-2])*tf.ones([d], dtype)
        basis = tf.constant([(-.5)**i for i in range(d)], dtype)
        p = tf.constant([.6+.4*i/(d-1) for i in range(d)], dtype)
        return dict(a=a, da=da, lower=lower, q=q, dq=dq, r=r, dr=r*v[-2],
                    m=theta[-1]*basis, dm=v[-1]*basis, p=p, h=tf.eye(d,dtype=dtype))

    def kalman_parameters(self, theta):
        p = self.parts(theta)
        return dict(transition_matrix=p['a'], process_covariance=p['q'],
                    observation_matrix=p['h'], observation_covariance=tf.linalg.diag(p['r']),
                    initial_mean=p['m'], initial_covariance=tf.linalg.diag(p['p']))

    def initial_cloud(self, theta, normals, direction=None):
        p = self.parts(theta, direction)
        shape = [tf.shape(normals)[0], self.dimension, self.dimension]
        states = p['m']+normals*tf.sqrt(p['p'])
        cov = tf.broadcast_to(tf.linalg.diag(p['p']), shape)
        return states, cov, tf.broadcast_to(p['dm'], tf.shape(states)), tf.zeros_like(cov)

    def model(self, theta, direction=None):
        active = [tf.zeros_like(theta) if direction is None else direction]
        fixed = self.parts(theta)
        def set_direction(value):
            active[0] = tf.cast(value, theta.dtype)
        def parts(value):
            return self.parts(value, active[0])
        def transition(value, points):
            return tf.linalg.matmul(points, parts(value)['a'], transpose_b=True)
        def transition_tangent(value, points, dpoints):
            p = parts(value)
            return (tf.linalg.matmul(dpoints, p['a'], transpose_b=True) +
                    tf.linalg.matmul(points, p['da'], transpose_b=True))
        def transition_density(value, points, means):
            p = parts(value)
            white = tf.linalg.triangular_solve(p['lower'], tf.transpose(points-means))
            return -.5*(tf.reduce_sum(tf.square(white), axis=0)+
                        2.*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(p['lower'])))+
                        tf.constant(self.dimension*math.log(2.*math.pi), theta.dtype))
        def transition_density_tangent(value, points, means, dpoints, dmeans):
            p = parts(value)
            inverse_residual = tf.transpose(tf.linalg.cholesky_solve(p['lower'], tf.transpose(points-means)))
            # d log N(r;Q) = -r^T Q^-1 dr + (r^T Q^-1 dQ Q^-1 r - tr(Q^-1 dQ))/2.
            quad = tf.reduce_sum(inverse_residual*tf.linalg.matmul(inverse_residual,p['dq']),axis=-1)
            trace = tf.linalg.trace(tf.linalg.cholesky_solve(p['lower'],p['dq']))
            return -tf.reduce_sum(inverse_residual*(dpoints-dmeans),axis=-1)+.5*(quad-trace)
        def observation_density(value, points, obs):
            p = parts(value)
            return -.5*tf.reduce_sum(tf.square(obs-points)/p['r']+tf.math.log(p['r'])+
                                     tf.constant(math.log(2.*math.pi),theta.dtype),axis=-1)
        def observation_density_tangent(value, points, obs, dpoints):
            p = parts(value)
            residual = obs-points
            return tf.reduce_sum(residual*dpoints/p['r']+
                                 .5*(tf.square(residual)/tf.square(p['r'])-1./p['r'])*p['dr'],axis=-1)
        return NonlinearScoreModel(
            transition_mean_fn=transition, transition_mean_tangent_fn=transition_tangent,
            observation_fn=lambda points: points,
            observation_jacobian_fn=lambda points: tf.broadcast_to(fixed['h'],[tf.shape(points)[0],self.dimension,self.dimension]),
            observation_tangent_fn=lambda points,dpoints: dpoints,
            process_covariance=fixed['q'], observation_covariance=tf.linalg.diag(fixed['r']),
            process_covariance_tangent_fn=lambda value: parts(value)['dq'],
            observation_covariance_tangent_fn=lambda value: tf.linalg.diag(parts(value)['dr']),
            transition_log_density_fn=transition_density, transition_log_density_tangent_fn=transition_density_tangent,
            observation_log_density_fn=observation_density, observation_log_density_tangent_fn=observation_density_tangent), set_direction

    def _simulate_tf(self, theta, horizon, seed):
        p, d = self.parts(theta), self.dimension
        noise = tf.random.stateless_normal([2*horizon+1,d],[seed,719],dtype=theta.dtype)
        state = p['m']+tf.sqrt(p['p'])*noise[0]
        observations = tf.TensorArray(theta.dtype,size=horizon)
        def body(t,state,observations):
            state = tf.linalg.matvec(p['a'],state)+tf.linalg.matvec(p['lower'],noise[2*t+1])
            obs = state+tf.sqrt(p['r'])*noise[2*t+2]
            return t+1,state,observations.write(t,obs)
        _,_,observations = tf.while_loop(lambda t,*_: t<horizon,body,(0,state,observations))
        return observations.stack()
