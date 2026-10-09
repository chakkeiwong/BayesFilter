"""Pure standard-library benchmark specifications; no data sampling or devices."""
import math
import random


def fixed_specification(name):
    if name == 'gaussian':
        return {'kind':'gaussian','mean':[1.,-1.],'covariance':[[13.,-12.],[-12.,13.]]}
    if name not in ('mixture','warped_mixture'):
        raise ValueError('unknown fixed control')
    return {'kind':'isotropic_mixture','dimension':2,'centers':[[-5.,0.],[5.,0.]],
            'variances':[1.,1.],'weights':[1/3,2/3],
            'warp_curvature':.1 if name == 'warped_mixture' else 0.,'warp_center':26.}


def random_mixture_specification(components, seed, *, distance=(6.,10.), variance=(.5,2.), translation=2.):
    """Frozen D=2 design choices from the October 3 benchmark plan.

    Standard-library RNG generates a target specification, not posterior data.
    Pairwise distances are independent uniform draws for the three-center case.
    """
    if components not in (2,3) or not 0 < distance[0] < distance[1] < 2*distance[0]:
        raise ValueError('two/three centers and strict triangle-compatible distance bounds required')
    if not 0 < variance[0] <= variance[1] or not math.isfinite(translation) or translation < 0:
        raise ValueError('invalid mixture design bounds')
    rng = random.Random(seed)
    a = rng.uniform(*distance)
    if components == 2:
        points = [(-a/2,0.),(a/2,0.)]
    else:
        b,c = rng.uniform(*distance),rng.uniform(*distance)
        u = (a*a+b*b-c*c)/(2*a)
        v = math.sqrt(b*b-u*u)
        points = [(0.,0.),(a,0.),(u,v)]
        center = [sum(row[j] for row in points)/3 for j in range(2)]
        points = [(x-center[0],y-center[1]) for x,y in points]
    angle = rng.uniform(0.,2*math.pi)
    cosine,sine = math.cos(angle),math.sin(angle)
    shift = [rng.uniform(-translation,translation) for _ in range(2)]
    points = [[cosine*x-sine*y+shift[0],sine*x+cosine*y+shift[1]] for x,y in points]
    exponentials = [-math.log1p(-rng.random()) for _ in range(components)]
    total = sum(exponentials)
    weights = [.1+(1-.1*components)*v/total for v in exponentials]
    return {'kind':'isotropic_mixture','dimension':2,'centers':points,
            'variances':[rng.uniform(*variance) for _ in points],'weights':weights,
            'warp_curvature':0.,'warp_center':0.,'design_seed':seed,
            'design_bounds':{'distance':list(distance),'variance':list(variance),'translation':translation}}
