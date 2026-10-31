"""Independent M10 posterior integration and analytical omission bounds.

Diagnostic/reference-only CPU TensorFlow arithmetic. No HMC or DCTR admission.
"""
from functools import lru_cache
import math

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import quadrature_rule
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import PRIOR_SD, source_theta
from bayesfilter.score_study.gaussian_tf import parameterized_model

DTYPE = tf.float64
RADIUS = 9*PRIOR_SD
MARGINS = (.1,)*5+(.02,)*6+(.03,)*2
FUNCTION_NAMES = ('standardized_log_q','standardized_log_r','centered_log_q_second',
    'centered_log_r_second','standardized_cross','log_q_cdf_q10','log_q_cdf_q50',
    'log_q_cdf_q90','log_r_cdf_q10','log_r_cdf_q50','log_r_cdf_q90',
    'replicated_y2_le_0.5','replicated_y2_le_2')


def polynomials(x,degree):
    values=[tf.ones_like(x),x]
    for k in range(1,degree):
        values.append(((2*k+1)*x*values[-1]-k*values[-2])/(k+1))
    return tf.stack(values[:degree+1],axis=-1)


def cdf_at(coefficients,points):
    """Each leading row holds Legendre coefficients of a normalized marginal."""
    degree=coefficients.shape[-1]-1
    values=polynomials(points,degree+1)
    antiderivatives=tf.concat([(points+1)[...,None],
        (values[...,2:]-values[...,:-2])/tf.cast(tf.range(1,degree+1)*2+1,DTYPE)],-1)
    return tf.reduce_sum(coefficients[:,None,:]*antiderivatives,axis=-1)


@lru_cache(maxsize=3)
def make_summary(order):
    if order not in (24,40,64):
        raise ValueError('unplanned posterior quadrature order')
    nodes,weights=quadrature_rule(order,normal=False)
    axis=nodes*RADIUS
    quadrature=weights[:,None]*weights[None,:]*RADIUS**2
    q,r=tf.meshgrid(axis,axis,indexing='ij')
    prior=tf.exp(-.5*(q*q+r*r)/PRIOR_SD**2)/(2*math.pi*PRIOR_SD**2)
    basis=polynomials(nodes,order-1)
    coefficient_scale=tf.cast(tf.range(order)*2+1,DTYPE)/2

    @tf.function(input_signature=[tf.TensorSpec([order,order],DTYPE),
        tf.TensorSpec([order,order,2],DTYPE),tf.TensorSpec([2],DTYPE),
        tf.TensorSpec([2],DTYPE),tf.TensorSpec([2,3],DTYPE)],jit_compile=True,autograph=False)
    def evaluate(likelihood,predictive,means,scales,cuts):
        mass=likelihood*prior*quadrature
        evidence=tf.reduce_sum(mass)
        mass/=evidence
        estimated_means=tf.stack([tf.reduce_sum(mass*q),tf.reduce_sum(mass*r)])
        estimated_scales=tf.sqrt(tf.stack([tf.reduce_sum(mass*(q-estimated_means[0])**2),
                                          tf.reduce_sum(mass*(r-estimated_means[1])**2)]))
        standardized=tf.stack([(q-means[0])/scales[0],(r-means[1])/scales[1]],axis=-1)
        first=tf.reduce_sum(mass[...,None]*standardized,axis=[0,1])
        second=tf.reduce_sum(mass[...,None]*standardized**2,axis=[0,1])
        cross=tf.reduce_sum(mass*standardized[...,0]*standardized[...,1])
        # Integrate out the other coordinate; density below is with respect
        # to normalized Legendre coordinates on [-1,1].
        marginal=tf.stack([tf.reduce_sum(mass,axis=1)/weights,
                           tf.reduce_sum(mass,axis=0)/weights])
        coefficients=tf.matmul(marginal*weights[None,:],basis)*coefficient_scale
        lower,upper=-tf.ones([2,3],DTYPE),tf.ones([2,3],DTYPE)
        probabilities=tf.constant([[.1,.5,.9],[.1,.5,.9]],DTYPE)
        def bisect(i,lo,hi):
            mid=(lo+hi)/2
            below=cdf_at(coefficients,mid)<probabilities
            return i+1,tf.where(below,mid,lo),tf.where(below,hi,mid)
        _,lower,upper=tf.while_loop(lambda i,*_:i<60,bisect,(0,lower,upper),parallel_iterations=1)
        quantiles=(lower+upper)*RADIUS/2
        cut_values=cdf_at(coefficients,cuts/RADIUS)
        predictions=tf.reduce_sum(predictive*(prior*quadrature)[...,None],axis=[0,1])/evidence
        expectations=tf.concat([first,second,[cross],tf.reshape(cut_values,[-1]),predictions],0)
        check_points=tf.broadcast_to(tf.linspace(tf.constant(-1.,DTYPE),tf.constant(1.,DTYPE),8193)[None,:],[2,8193])
        cdf=cdf_at(coefficients,check_points)
        density=tf.einsum('ik,ijk->ij',coefficients,polynomials(check_points,order-1))
        interpolation_valid=(tf.reduce_min(cdf)>=-1e-8)&(tf.reduce_max(cdf)<=1+1e-8)
        interpolation_valid &= tf.reduce_min(cdf[:,1:]-cdf[:,:-1])>=-1e-8
        interpolation_valid &= tf.reduce_min(density)>=-1e-5*tf.reduce_max(density)
        return dict(evidence=evidence,means=estimated_means,scales=estimated_scales,
            cuts=quantiles,expectations=expectations,interpolation_valid=interpolation_valid,
            minimum_cdf_increment=tf.reduce_min(cdf[:,1:]-cdf[:,:-1]),
            minimum_interpolated_density=tf.reduce_min(density),coefficients=coefficients)
    return axis,evaluate


def normal_probability(lower,upper,mean=0.,sd=1.):
    return .5*(math.erfc((lower-mean)/(sd*math.sqrt(2)))-
               math.erfc((upper-mean)/(sd*math.sqrt(2))))


def evidence_lower_bound(observations):
    """An explicit positive sub-integral, independent of posterior quadrature."""
    a,_,h,_,m,_,p,_,q,_,r,_=parameterized_model(source_theta(tf.zeros([2],DTYPE)),1,1)
    a,h,m,p,q,r=map(float,(a[0,0],h[0,0],m[0],p[0,0],q[0,0],r[0,0]))
    c,b=float(tf.cast(.2,DTYPE)),float(tf.cast(.12,DTYPE))
    data=[float(x) for x in tf.reshape(observations,[-1])]
    roots=[(-h+math.sqrt(max(0.,h*h+4*b*y)))/(2*b) for y in data]
    def f(x): return a*x+c*math.sin(x)
    def obs(x): return h*x+b*x*x
    def lower_density(residual,sd_low,sd_high):
        # 1/sd >= 1/sd_high and exp(-e^2/(2 sd^2)) >= exp(-E^2/(2 sd_low^2)).
        return math.exp(-.5*(residual/sd_low)**2)/(math.sqrt(2*math.pi)*sd_high)
    best=dict(lower_bound=0.)
    for half in (.05,.1,.2,.4):
        states=[(m-half,m+half),*( (x-half,x+half) for x in roots)]
        p0=normal_probability(*states[0],m,math.sqrt(p))
        residual_q=[max(abs(right-f(left)) for left in states[i] for right in states[i+1]) for i in range(2)]
        residual_r=[]
        for y,(lo,hi) in zip(data,states[1:]):
            values=[obs(lo),obs(hi)]
            vertex=-h/(2*b)
            if lo<=vertex<=hi: values.append(obs(vertex))
            residual_r.append(max(abs(y-x) for x in values))
        for qi in range(-2,6):
            qlo,qhi=qi*PRIOR_SD/2,(qi+1)*PRIOR_SD/2
            sqlo,sqhi=math.sqrt(q)*math.exp(qlo),math.sqrt(q)*math.exp(qhi)
            for ri in range(-2,6):
                rlo,rhi=ri*PRIOR_SD/2,(ri+1)*PRIOR_SD/2
                srlo,srhi=math.sqrt(r)*math.exp(rlo),math.sqrt(r)*math.exp(rhi)
                bound=(p0*(2*half)**2*normal_probability(qlo,qhi,sd=PRIOR_SD)*
                       normal_probability(rlo,rhi,sd=PRIOR_SD))
                for e in residual_q: bound*=lower_density(e,sqlo,sqhi)
                for e in residual_r: bound*=lower_density(e,srlo,srhi)
                if bound>best['lower_bound']:
                    best=dict(lower_bound=bound,states=states,log_q_interval=[qlo,qhi],
                        log_r_interval=[rlo,rhi],transition_residual_bounds=residual_q,
                        observation_residual_bounds=residual_r,initial_mass=p0)
    return best


def omission_allowance(means,scales,expectations,evidence_lower):
    """Analytical union bounds; no quadrature evidence used as denominator."""
    s=PRIOR_SD
    tilt_means=(0.,-2*s*s)
    coefficient=math.exp(1.2+2*s*s)/(2*math.pi)
    probabilities=[]
    tail_seconds=[]
    full_seconds=[]
    for t,m,scale in zip(tilt_means,means,scales):
        lo,hi=(-RADIUS-t)/s,(RADIUS-t)/s
        p_left=.5*math.erfc(-lo/math.sqrt(2))
        p_right=.5*math.erfc(hi/math.sqrt(2))
        phi_lo=math.exp(-lo*lo/2)/math.sqrt(2*math.pi)
        phi_hi=math.exp(-hi*hi/2)/math.sqrt(2*math.pi)
        prob=p_left+p_right
        first=phi_hi-phi_lo
        second=prob+hi*phi_hi-lo*phi_lo
        tail_seconds.append(((t-m)**2*prob+2*(t-m)*s*first+s*s*second)/scale**2)
        full_seconds.append(((t-m)**2+s*s)/scale**2)
        probabilities.append(prob)
    prob_bound=sum(probabilities)
    second_bounds=[tail_seconds[i]+full_seconds[i]*probabilities[1-i] for i in range(2)]
    state_prob=3*math.erfc(10/math.sqrt(2))
    # Include state truncation over all parameters using its uniform conditional
    # probability bound and the same envelope, also for unbounded moments.
    second_bounds=[b+state_prob*f for b,f in zip(second_bounds,full_seconds)]
    prob_bound+=state_prob
    first_bounds=[math.sqrt(prob_bound*b) for b in second_bounds]
    absolute=first_bounds+second_bounds+[sum(second_bounds)/2]+[prob_bound]*8
    allowances=[coefficient*(b+abs(e)*prob_bound)/evidence_lower for b,e in zip(absolute,expectations)]
    return dict(likelihood_envelope_coefficient=coefficient,evidence_lower_bound=evidence_lower,
        unnormalized_omitted_mass_bound=coefficient*prob_bound,
        posterior_omitted_mass_bound=coefficient*prob_bound/evidence_lower,
        function_allowances=allowances,parameter_tail_probability_union=sum(probabilities),
        state_omission_probability_bound=state_prob)
