"""Independent high-precision diagnostic of the v7 float64 inversion envelope.

The reference uses Python Decimal, not the TensorFlow formula or its guard.
Tests establish the named examples/device, not a formal hardware error theorem.
"""
from collections import Counter
from decimal import Decimal, localcontext
import math
import os

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_statistics import bounded_trial_intervals


BETTING = "bounded_betting_mixture_v1"
HOEFFDING = "bounded_hoeffding_rungs_v1"
BETS = (1., .5, .25, .125)


def reference_interval(values, alpha, method):
    """80-digit independent root inversion, preserving exact float inputs."""
    with localcontext() as context:
        context.prec = 80
        counts = [(Decimal.from_float(x), count) for x, count in Counter(values).items()]
        n = len(values)
        eta = Decimal.from_float(alpha)
        if method == HOEFFDING:
            mean = sum(x * count for x, count in counts) / n
            half = ((-eta.ln()) / (2*n)).sqrt()
            return max(Decimal(0), mean-half), min(Decimal(1), mean+half)
        threshold = 1/eta

        def capital(mu, sign):
            total = Decimal(0)
            for b in BETS:
                bet = Decimal.from_float(b)
                product = Decimal(1)
                for value, count in counts:
                    product *= (1 + sign*bet*(value-mu))**count
                total += product
            return total / len(BETS)

        bounds = []
        for sign in (1, -1):
            left, right = Decimal(0), Decimal(1)
            for _ in range(180):
                mid = (left+right)/2
                crossed = capital(mid, sign) > threshold
                if (sign == 1 and crossed) or (sign == -1 and not crossed):
                    left = mid
                else:
                    right = mid
            bounds.append((left+right)/2)
        return tuple(bounds)


def _near_crossing_score(threshold, n, alpha, sign):
    """Construct a constant-score input whose exact boundary nearly touches a gate."""
    with localcontext() as context:
        context.prec = 80
        left, right = Decimal(0), Decimal(1)
        eta = Decimal.from_float(alpha)
        mu = Decimal.from_float(threshold)
        for _ in range(180):
            mid = (left+right)/2
            capital = sum((1+sign*Decimal.from_float(b)*(mid-mu))**n for b in BETS)/len(BETS)
            if (sign == 1 and capital > 1/eta) or (sign == -1 and capital <= 1/eta):
                right = mid
            else:
                left = mid
        return float((left+right)/2)


@pytest.mark.parametrize("n,alpha",[(64,.05/(2*2*100*5)),(1024,.05/(2*3*100*24)),(16384,1e-12)])
@pytest.mark.parametrize("method",[BETTING,HOEFFDING])
def test_endpoints_and_decision_boundaries_enclose_high_precision_reference(n,alpha,method):
    columns = [[0.]*n, [1.]*n, [0.,1.]*(n//2), [.001,.999]*(n//2)]
    for threshold in (.55,.65,.75,.85):
        sign = 1 if threshold < .7 else -1
        value = _near_crossing_score(threshold,n,alpha,sign)
        columns.extend([[math.nextafter(value,direction)]*n for direction in (0.,1.)])
    gpu = os.environ.get("BAYESFILTER_REQUIRE_GPU_VALIDATION") == "1"
    device = "/GPU:0" if gpu else "/CPU:0"
    if gpu:
        assert tf.config.list_logical_devices("GPU"), "trusted GPU evidence cannot fall back to CPU"
    with tf.device(device):
        lo,hi = bounded_trial_intervals(list(map(list,zip(*columns))),sided_alpha=alpha,
            method=method,bets=BETS,jit_compile=True)
        assert device.replace('/','') in lo.device and device.replace('/','') in hi.device
    for index, values in enumerate(columns):
        lower,upper = float(lo[index]),float(hi[index])
        assert 0 <= lower <= upper <= 1
        ref_low,ref_high = reference_interval(values,alpha,method)
        # 2**-180 is the reference bracket, not a tolerance for inward error.
        reference_width = Decimal(2)**-180
        assert Decimal.from_float(lower) <= ref_low+reference_width, (n,index,lower,str(ref_low))
        assert Decimal.from_float(upper) >= ref_high-reference_width, (n,index,upper,str(ref_high))


def test_reference_distinguishes_inward_mutation_and_threshold_ambiguity():
    values = [.7]*256
    lower,upper = reference_interval(values,.000025,BETTING)
    assert Decimal.from_float(float(lower)+1e-8) > lower
    assert Decimal.from_float(float(upper)-1e-8) < upper
    for threshold,sign in ((.55,1),(.85,-1)):
        value = _near_crossing_score(threshold,256,.000025,sign)
        bounds = [reference_interval([math.nextafter(value,d)]*256,.000025,BETTING)
                  for d in (0.,1.)]
        index = 0 if sign == 1 else 1
        assert bounds[0][index] < Decimal.from_float(threshold) < bounds[1][index]
