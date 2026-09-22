"""Independent CPU diagnostic calibration; no tuning or default promotion.

Stan comparator is an independent translation of posterior R/convergence.R
.ess lines 732--790, using its N-lag autocovariance and initial-monotone rule.
NumPy/SciPy here are independent references, never runtime inference inputs.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm, t as student_t
import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.inference.hmc_precision import mean_precision


def stan_ess_reference(x):
    """[draw, chain, replicate]; source-matching vectorized independent reference."""
    n, m, p = x.shape
    y = x - x.mean(axis=0)
    size = 1 << (2*n-1).bit_length()
    spectrum = np.fft.rfft(y, n=size, axis=0)
    ac = np.fft.irfft(spectrum * spectrum.conj(), n=size, axis=0)[:n]
    ac = (ac / np.arange(n,0,-1)[:,None,None]).mean(axis=1)
    w = ac[0] * n/(n-1)
    vp = ac[0] + x.mean(axis=0).var(axis=0,ddof=1)
    result = np.empty(p)
    for column in range(p):
        rho = np.zeros(n)
        lag = 0
        even = 1.
        odd = 1-(w[column]-ac[1,column])/vp[column]
        rho[:2] = [even,odd]
        while lag < n-5 and np.isfinite(even+odd) and even+odd > 0:
            lag += 2
            even = 1-(w[column]-ac[lag,column])/vp[column]
            odd = 1-(w[column]-ac[lag+1,column])/vp[column]
            if even+odd >= 0:
                rho[lag:lag+2] = [even,odd]
        maximum = lag
        if even > 0:
            rho[maximum] = even
        lag = 0
        while lag <= maximum-4:
            lag += 2
            previous = rho[lag-2]+rho[lag-1]
            if rho[lag]+rho[lag+1] > previous:
                rho[lag:lag+2] = previous/2
        # In R, 1:0 yields c(1,0), so the first element remains in the sum.
        tau = -1+2*rho[:max(maximum,1)].sum()+rho[maximum]
        result[column] = m*n/max(tau,1/np.log10(m*n))
    return result


def wilson(success, count):
    z = norm.ppf(.975)
    estimate = success/count
    center = (estimate+z*z/(2*count))/(1+z*z/count)
    radius = z*np.sqrt(estimate*(1-estimate)/count+z*z/(4*count*count))/(1+z*z/count)
    return [float(center-radius),float(center+radius)]


def mean_ci(x):
    v = np.asarray(x)
    half = student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return [float(v.mean()-half),float(v.mean()+half)]


def main():
    root = Path(__file__).resolve().parent
    started = time.monotonic()
    rng = np.random.default_rng(20260915)
    rows = []
    for rho in (0., .8, -.5):
        n, chains, replications = 2048, 4, 200
        noise = rng.normal(size=(n,chains,replications))
        noise[1:] *= np.sqrt(1-rho*rho)
        values = lfilter([1.],[1.,-rho],noise,axis=0)
        errors = values.mean(axis=(0,1))
        truth_mcse = np.sqrt((1+rho)/(1-rho)/(n*chains))
        sd = values.reshape(-1,replications).std(axis=0,ddof=1)
        half = n//2
        split = np.concatenate((values[:half],values[-half:]),axis=1)
        estimators = {}
        for method in ('autocorrelation','batch_means','lugsail'):
            report = mean_precision(tf.constant(values),method=method,jit_compile=False)
            estimators[method] = np.asarray(report['mcse'])
        estimators['stan_initial_monotone_reference'] = sd/np.sqrt(stan_ess_reference(split))
        for method, mcse in estimators.items():
            valid = np.isfinite(mcse)&(mcse>0)
            covered = valid & (np.abs(errors) <= norm.ppf(.975)*mcse)
            ratio = mcse[valid]/truth_mcse
            rows.append({'rho':rho,'method':method,'replications':replications,
                'valid_replications':int(valid.sum()), 'truth_mean_mcse':float(truth_mcse),
                'empirical_mean_sd':float(errors.std(ddof=1)),
                'mean_estimated_mcse':float(mcse[valid].mean()),
                'mean_mcse_to_truth':float(ratio.mean()),'mean_ratio_ci95':mean_ci(ratio),
                'normal_interval_coverage':float(covered.mean()),
                'coverage_wilson95':wilson(int(covered.sum()),replications),
                'interpretation':'fixed-size coverage diagnostic, not sequential stopping coverage'})
    body = {'rows':rows,'seed':20260915,'data':'stationary unit-variance Gaussian AR(1)',
        'jit_compile':False,'gpu_intentionally_hidden':True,
        'runtime_backend':'TensorFlow 2.20/TFP 0.25; NumPy/SciPy independent comparator only',
        'baseline':'known AR(1) LRV and realized replicate means; four estimators on identical draws',
        'nonclaims':['no statistical superiority ranking','no burn-in calibration',
            'no arbitrary-target or heavy-tail guarantee','no sequential or simultaneous interval coverage'],
        'wall_seconds':time.monotonic()-started}
    (root/'precision-calibration.json').write_text(json.dumps(body,indent=2,allow_nan=False)+'\n')
    sources = [Path(__file__),ROOT/'bayesfilter/inference/hmc_precision.py',
        ROOT/'bayesfilter/inference/hmc_diagnostic_math.py',
        ROOT/'.localresources/papers/hmc_warmup_precision_20260915/posterior-convergence.R']
    manifest = {'command':sys.argv,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'environment':{'python':sys.version,'tensorflow':tf.__version__,'tfp':tfp.__version__,'conda_env':'tfgpu'},
        'cpu_gpu':'CPU-only; GPU intentionally hidden before import','jit_compile':False,
        'jit_exception':'independent-reference calibration; not HMC execution or default evidence',
        'data_version':'stationary_AR1_2048x4x200_v1','seed':20260915,'wall_seconds':body['wall_seconds'],
        'plan':'docs/plans/bayesfilter-hmc-overall-repair-plan-2026-09-15.md',
        'result':str(root/'precision-calibration.json'),
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (root/'calibration-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'completed','rows':len(rows),'wall_seconds':body['wall_seconds']}))


if __name__ == '__main__':
    main()
