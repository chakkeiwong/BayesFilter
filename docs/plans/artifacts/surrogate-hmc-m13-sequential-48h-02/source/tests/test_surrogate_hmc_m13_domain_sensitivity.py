"""Analytic independent checks of the positive-weight domain extension."""
import math
import pytest
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_m13_domain_sensitivity import (
    PLAN, PRIOR_SD, quadrature_design, integrator,
)
from bayesfilter.inference.surrogate_hmc_closure_program import jobs, PASS_CAPS


def test_shell_quadrature_geometry_and_polynomial_integrals():
    keys,points,w,shell=quadrature_design()
    assert len(keys)==len(set(keys))==129**2+2336
    assert sum(abs(x)>64 or abs(y)>64 for x,y in keys)==2336
    assert min(w)>0 and min(shell)>=0
    radius=5*PRIOR_SD
    assert math.fsum(w)==pytest.approx((2*radius)**2,rel=1e-14)
    assert math.fsum(shell)==pytest.approx((10*PRIOR_SD)**2-(8*PRIOR_SD)**2,rel=1e-14)
    for dimension in (0,1):
        assert math.fsum(a*p[dimension] for a,p in zip(w,points))==pytest.approx(0.,abs=1e-14)
        assert math.fsum(a*p[dimension]**2 for a,p in zip(w,points))==pytest.approx((2*radius)**2*radius**2/3,rel=1e-14)


def test_analytic_gaussian_prior_domain_mass_and_normalization_influence():
    _,points,w,shell=quadrature_design()
    evaluate=integrator(points,w,shell,4)
    logs=tf.broadcast_to(tf.constant([-.5,0.,.2,.4],tf.float64),[len(points),4])
    result=evaluate(logs)
    z5=math.erf(5/math.sqrt(2));z4=math.erf(4/math.sqrt(2))
    sd=PRIOR_SD*math.sqrt(1-10*math.exp(-12.5)/math.sqrt(2*math.pi)/z5)
    assert bool(result['finite'])
    # Independent separable 1D Simpson calculation verifies the actual rule.
    def integrate_1d(radius,step,power=0):
        n=round(2*radius/step)
        return math.fsum(((-radius+i*step)**power)*math.exp(-(-radius+i*step)**2/(2*PRIOR_SD**2))
            /(PRIOR_SD*math.sqrt(2*math.pi))*(1 if i in (0,n) else 4 if i%2 else 2)
            for i in range(n+1))*step/3
    zf=integrate_1d(4*PRIOR_SD,PRIOR_SD/16)
    zi=integrate_1d(4*PRIOR_SD,PRIOR_SD/8)
    zo=integrate_1d(5*PRIOR_SD,PRIOR_SD/8)
    mass=zf*zf+zo*zo-zi*zi
    second=(integrate_1d(4*PRIOR_SD,PRIOR_SD/16,2)*zf
            +integrate_1d(5*PRIOR_SD,PRIOR_SD/8,2)*zo
            -integrate_1d(4*PRIOR_SD,PRIOR_SD/8,2)*zi)/mass
    assert result['metrics'].numpy().tolist()==pytest.approx(
        [0.,0.,math.sqrt(second),math.sqrt(second),(zo*zo-zi*zi)/mass],rel=0,abs=2e-14)
    # Coarse-shell quadrature error is distinct from implementation error.
    assert result['metrics'][:4].numpy().tolist()==pytest.approx([0.,0.,sd,sd],rel=0,abs=1e-7)
    assert float(result['metrics'][4])==pytest.approx(1-(z4/z5)**2,rel=0,abs=1e-7)
    assert float(tf.reduce_max(result['mcse']))<1e-14
    bad=tf.tensor_scatter_nd_update(logs,[[0,0]],tf.constant([float('nan')],tf.float64))
    assert not bool(evaluate(bad)['finite'])


def test_master_domain_registration():
    selected=jobs('m13-domain-sensitivity')
    assert [x['name'] for x in selected]==['review','m13-domain-sensitivity']
    assert selected[1]['plan']==PLAN and selected[1]['device']=='gpu'
    assert PASS_CAPS['m13-domain-sensitivity']==dict(cpu_worker_seconds=1000.,gpu_seconds=1100.)
