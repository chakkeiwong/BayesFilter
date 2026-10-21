"""Model laws for the shared batch-native analytical covariance proposal.

Values have leading B; tangents have leading P,B. No sample mapping or
autodiff is used. Scalar model callbacks remain independent parity references.
This module supplies model laws only; proposal, filtering and reset mathematics
have one authority in covariance_proposal_score_tf / score_batch_tf.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, exp, log, sqrt, logsumexp, stack, mm, mv, solve


def diagonal(x):
    return Jet(tf.linalg.diag(x.v), tf.linalg.diag(x.t))


class NativeModels:
    """Native B-row model laws and analytical partials for all P parameters.

    Shared Jet operations reuse the primal. The range-bearing local polar
    callback currently flattens P,B,N for its analytic directional interface;
    that bounded callback still repeats primal work across parameter directions.
    """
    def __init__(self, spec, theta):
        self.spec = spec
        self.batch, self.p = theta.shape
        self.d = spec.dimension
        self.theta = theta
        self.q = Jet(theta, tf.broadcast_to(tf.eye(self.p, dtype=theta.dtype)[:, None],
                                          [self.p, self.batch, self.p]))
        self.kind = getattr(spec, 'name', getattr(spec, 'family', None))
        if self.kind is None and hasattr(spec, 'spec') and spec.spec.get('kind') == 'm13':
            self.kind = 'range_bearing'
        if self.kind not in ('sir_d18', 'predator_prey', 'ksc_mixture', 'p44',
                             'diagonal_ar', 'frozen_3d', 'full_matrix', 'range_bearing'):
            raise ValueError('No native model adapter for ' + str(self.kind))
        self.constant_observation_jacobian = self.kind != 'range_bearing'
        self._prepare()

    def c(self, x):
        return self.q.constant(x)

    def _prepare(self):
        q, d, b, dtype = self.q, self.d, self.batch, self.theta.dtype
        eye = self.c(tf.eye(d, batch_shape=[b], dtype=dtype))
        self.H = eye
        self.mean = self.c(tf.zeros([b, d], dtype))
        self.initial_scale = self.c(tf.ones([b, d], dtype))
        if self.kind in ('sir_d18', 'predator_prey'):
            self.mean = self.c(self.spec.initial_mean(dtype)).broadcast([b, d])
            if self.kind == 'sir_d18':
                from bayesfilter.highdim.models import _zhao_cui_sir_austria_adjacency_xla
                self.adjacency = self.c(tf.cast(_zhao_cui_sir_austria_adjacency_xla(), dtype))
                self.degree = self.c(tf.reduce_sum(self.adjacency.v, -1))
                self.H = self.c(tf.one_hot(2*tf.range(9)+1, 18, dtype=dtype)).broadcast([b, 9, 18])
                self.Q = eye
                self.R = exp(2*q[:, 2, None, None])*100*self.c(tf.eye(9, dtype=dtype))
            else:
                self.Q = 4*eye
                self.R = 4*eye
        elif self.kind == 'ksc_mixture':
            # Exact constants and moment proposal from ksc_sv_canonical_model.
            self.weights = self.c(tf.constant([.00730,.10556,.00002,.04395,.34001,.24566,.25750], dtype))
            self.means = self.c(tf.constant([-10.12999,-3.97281,-8.56686,2.77786,.61942,1.79518,-1.08819], dtype)-tf.constant(1.2704,dtype))
            self.variances = self.c(tf.constant([5.79596,2.61369,5.17950,.16735,.64009,.34023,1.26261],dtype))
            self.mixmean = (self.weights*self.means).sum(-1)
            mixvar = (self.weights*(self.variances+self.means*self.means)).sum(-1)-self.mixmean*self.mixmean
            t = q[:, 0]
            self.phi = Jet(.5*(1+tf.math.erf(t.v/tf.constant(math.sqrt(2),dtype))),
                           tf.exp(-.5*t.v*t.v)*t.t/tf.constant(math.sqrt(2*math.pi),dtype))[:,None]
            self.Q, self.R = eye, mixvar*eye
        elif self.kind == 'range_bearing':
            fixture = self.spec.spec['fixture']
            self.A = self.c(tf.cast(fixture.A,dtype))
            self.Q = exp(2*q[:,0,None,None])*self.c(tf.cast(fixture.Q,dtype))
            self.R = exp(2*q[:,1,None,None])*self.c(tf.cast(fixture.R,dtype))
        else:
            def padded(first, rest):
                return self.c(tf.constant((first+[rest]*max(0,d-3))[:d],dtype))
            if self.kind == 'p44':
                t = q[:,0,None]
                tanh = Jet(tf.tanh(t.v),(1-tf.tanh(t.v)**2)*t.t)
                self.phi = .55*tanh*padded([1.,.85,.70],.55)
                self.Q = diagonal(exp(q[:,1,None])*padded([.9,1.1,1.3],1.))
                self.R = diagonal(exp(q[:,2,None])*padded([1.,1.2,.8],1.))
                self.mean = q[:,3,None]*padded([1.,-.5,.25],0.)
                self.initial_scale = sqrt(padded([.6,.8,1.],1.)).broadcast([b,d])
            elif self.kind == 'full_matrix':
                self.A = q[:,:d*d].reshape([b,d,d])
                indices = [(i,j) for i in range(d) for j in range(i+1)]
                entries = q[:,d*d:-2]
                mask = tf.constant([i==j for i,j in indices])
                ev = tf.where(mask,tf.exp(entries.v),entries.v)
                et = tf.where(mask,ev*entries.t,entries.t)
                basis = self.c(tf.reshape(tf.one_hot([i*d+j for i,j in indices],d*d,dtype=dtype),[len(indices),d,d]))
                lower = (Jet(ev,et)[:,:,None,None]*basis[None]).sum(1)
                self.Q = mm(lower,lower.transpose())
                self.R = exp(q[:,-2,None,None])*eye
                # Match the existing constant preparation exactly, outside hot arithmetic.
                with tf.init_scope():
                    index = tf.cast(tf.range(d),tf.float64)
                    sign = tf.where(tf.range(d)%2==0,tf.constant(1.,tf.float64),tf.constant(-1.,tf.float64))
                    basis_m = tf.cast(sign*tf.pow(tf.constant(.5,tf.float64),index),dtype)
                    initial_variance = tf.cast(.6+.4*index/(d-1),dtype)
                self.mean = q[:,-1,None]*self.c(basis_m)
                self.initial_scale = sqrt(self.c(initial_variance)).broadcast([b,d])
            else:
                self.phi = q[:,:d]
                self.Q = q[:,d,None,None]**2*eye
                self.R = q[:,d+1,None,None]**2*eye
                if self.kind == 'frozen_3d':
                    self.H = self.c(tf.constant([[1.,.25,-.15],[.2,1.1,.3],[-.1,.35,.9]],dtype)).broadcast([b,d,d])

    def initial_cloud(self, normals):
        if self.kind == 'range_bearing':
            return self.c(normals[None]).broadcast([self.batch,*normals.shape])
        return self.mean[:,None]+self.c(normals[None])*self.initial_scale[:,None]

    def _rhs(self, x):
        q = self.q
        if self.kind == 'sir_d18':
            s, i = x[...,0::2], x[...,1::2]
            kappa, nu = .1*exp(q[:,0,None,None]), 18*exp(q[:,1,None,None])
            infection = kappa*s*i
            ds = -infection+.5*(mm(s,self.adjacency.transpose())-s*self.degree)
            di = infection-nu*i+.5*(mm(i,self.adjacency.transpose())-i*self.degree)
            return stack([ds,di],-1).reshape(x.v.shape.as_list())
        r, capacity, half, s, u, v = [q[:,k,None] for k in range(6)]
        prey, predator = x[...,0], x[...,1]
        interaction = prey*predator/(half+prey)
        logistic = prey*(1-prey/capacity)
        return stack([r*logistic-s*interaction,u*interaction-v*predator],-1)

    def transition(self, x):
        if self.kind not in ('sir_d18','predator_prey'):
            return mv(self.A[:,None] if self.A.v.shape.rank==3 else self.A,x) if hasattr(self,'A') else x*self.phi[:,None]
        step = tf.constant(.005 if self.kind=='sir_d18' else .1,x.v.dtype)
        steps = 4 if self.kind=='sir_d18' else 20
        k4_scale = .5 if self.kind=='sir_d18' else 1.
        def body(k,v,t):
            state = Jet(v,t)
            k1 = self._rhs(state)
            k2 = self._rhs(state+.5*step*k1)
            k3 = self._rhs(state+.5*step*k2)
            k4 = self._rhs(state+k4_scale*step*k3)
            z = state+step/6*(k1+2*k2+2*k3+k4)
            return k+1,z.v,z.t
        _,v,t = tf.while_loop(lambda k,*_:k<steps,body,(0,x.v,x.t),parallel_iterations=1)
        return Jet(v,t)

    def _polar(self,x,jacobian=False):
        from bayesfilter.highdim.ledh_range_bearing_target_tf import (polar_observation,
            polar_observation_tangent,polar_jacobian,polar_jacobian_tangent)
        flat = tf.reshape(x.v,[-1,self.d])
        fn,dfn = (polar_jacobian,polar_jacobian_tangent) if jacobian else (polar_observation,polar_observation_tangent)
        v = fn(flat)
        # Local directional callbacks see a single native P*B*N tensor, not P replays.
        repeated = tf.broadcast_to(flat[None],[self.p,*flat.shape])
        t = dfn(tf.reshape(repeated,[-1,self.d]),tf.reshape(x.t,[-1,self.d]))
        shape = [*x.v.shape[:-1],*v.shape[1:]]
        return Jet(tf.reshape(v,shape),tf.reshape(t,[self.p,*shape]))

    def observation(self,x):
        if self.kind=='range_bearing': return self._polar(x)
        y = mv(self.H[:,None],x)
        if self.kind=='ksc_mixture': y = y+self.mixmean+2*self.q[:,1,None,None]
        return y

    def observation_density(self,x,y):
        if self.kind=='sir_d18':
            # Preserve the source density's finite FP32 expression, including
            # exp(log(100)+2*theta), independently of the proposal R matrix.
            # See ledh_canonical_models_tf.sir_austria_canonical_model.
            residual = self.c(y)-self.observation(x)
            log_variance = self.c(tf.math.log(tf.constant(100.,x.v.dtype)))+2*self.q[:,2,None]
            return -.5*((residual*residual).sum(-1)/exp(log_variance)
                         +9*(math.log(2*math.pi)+log_variance))
        if self.kind=='ksc_mixture':
            residual = self.c(y[0])-2*self.q[:,1,None]-x[...,0]
            r = residual[...,None]-self.means
            return logsumexp(log(self.weights)-.5*(r*r/self.variances+log(self.variances)+math.log(2*math.pi)),-1)
        if self.kind=='range_bearing':
            model = self.spec.spec['model']
            shape = x.v.shape.as_list(); count = shape[1]
            theta = tf.broadcast_to(self.theta[:,None],[self.batch,count,self.p])
            v = model.observation_log_density_fn(tf.reshape(theta,[-1,self.p]),tf.reshape(x.v,[-1,self.d]),y)
            theta_p = tf.broadcast_to(theta[None],[self.p,*theta.shape])
            states = tf.broadcast_to(x.v[None],[self.p,*shape])
            direction = tf.broadcast_to(self.q.t[:,:,None],[self.p,self.batch,count,self.p])
            t = model.observation_log_density_tangent_fn(tf.reshape(theta_p,[-1,self.p]),
                tf.reshape(states,[-1,self.d]),y,tf.reshape(x.t,[-1,self.d]),tf.reshape(direction,[-1,self.p]))
            return Jet(tf.reshape(v,shape[:-1]),tf.reshape(t,[self.p,*shape[:-1]]))
        residual = self.c(y)-self.observation(x)
        # All current Gaussian observation laws have diagonal R (full-matrix LGSSM included).
        variance = Jet(tf.linalg.diag_part(self.R.v),tf.linalg.diag_part(self.R.t))[:,None]
        return -.5*(residual*residual/variance+log(variance)+math.log(2*math.pi)).sum(-1)

    def apply(self,value_name,tangent_name,x,*args,theta=False):
        del tangent_name,theta
        if value_name=='transition_mean_fn': return self.transition(x)
        if value_name=='observation_fn': return self.observation(x)
        if value_name=='observation_jacobian_fn':
            return self._polar(x,True) if self.kind=='range_bearing' else self.H[:,None].broadcast([*x.v.shape[:-1],*self.H.v.shape[-2:]])
        if value_name=='observation_log_density_fn': return self.observation_density(x,*args)
        raise ValueError(value_name)

    def covariance(self,value_name,tangent_name):
        del tangent_name
        return self.Q if value_name=='process_covariance' else self.R
