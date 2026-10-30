#!/usr/bin/env python3
"""Independent NumPy/SciPy/Octave diagnostic of the pinned normal-quantile shim."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.special import ndtri, ndtr

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party/audit/zhao_cui_tensor_ssm_p10/source/octave_compat'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lower = np.unique(np.r_[np.nextafter(0., 1.), np.logspace(-300, -1, 101),
                            2. ** np.arange(-60, -49), .25, .5])
    p = np.r_[-.1, 0., lower, np.unique(1. - lower)[::-1], 1., 1.1, np.nan]
    np.savetxt(out / 'probabilities.csv', p, fmt='%.17g')
    q = lambda path: "'" + str(path).replace("'", "''") + "'"
    script = f"""addpath({q(SOURCE)});
p=dlmread({q(out / 'probabilities.csv')});
old=norminv(p); candidate=NaN(size(p));
lo=p>=0 & p<=.5; hi=p>.5 & p<=1;
candidate(lo)=-sqrt(2).*erfcinv(2.*p(lo));
candidate(hi)=sqrt(2).*erfcinv(2.*(1-p(hi)));
dlmwrite({q(out / 'octave-quantiles.csv')},[old,candidate],'precision','%.17g');
"""
    (out / 'run.m').write_text(script)
    start = time.monotonic()
    with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
        run = subprocess.run(['octave-cli', '--quiet', '--no-gui', str(out / 'run.m')],
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1',
                                  'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                             stdout=stdout, stderr=stderr, timeout=60)
    run.check_returncode()
    old, candidate = np.loadtxt(out / 'octave-quantiles.csv', delimiter=',').T
    reference = ndtri(p)
    interior = (p > 0) & (p < 1)
    finite = interior & np.isfinite(candidate)
    old_valid = interior & np.isfinite(old)
    # Relative CDF error is informative in the lower tail; absolute is not.
    moderate = interior & (p >= 1e-280) & (p <= .5)
    result = dict(schema='zhao_cui_normal_quantile_diagnostic_v1',
                  role='independent_reference_diagnostic', cpu_only=True,
                  gpu_intentionally_hidden=True, count=int(p.size),
                  old_nonfinite_at_valid_probabilities=int(np.sum(interior & ~np.isfinite(old))),
                  candidate_nonfinite_at_valid_probabilities=int(np.sum(interior & ~np.isfinite(candidate))),
                  maximum_candidate_error_vs_scipy=float(np.max(np.abs(candidate[finite]-reference[finite]))),
                  maximum_old_error_where_finite=float(np.max(np.abs(old[old_valid]-reference[old_valid]))),
                  maximum_lower_cdf_relative_roundtrip_error=float(np.max(np.abs(ndtr(candidate[moderate])/p[moderate]-1))),
                  domain_and_endpoint_semantics=bool(np.array_equal(np.isnan(candidate), np.isnan(reference))
                      and np.array_equal(np.isposinf(candidate), np.isposinf(reference))
                      and np.array_equal(np.isneginf(candidate), np.isneginf(reference))),
                  wall_seconds=time.monotonic()-start,
                  limitation='This checks the compatibility shim; it does not identify the live smoothing failure cause.')
    np.savetxt(out / 'comparison.csv', np.c_[p,old,candidate,reference], delimiter=',',
               header='probability,pinned_erfinv,stable_erfcinv,scipy_ndtri', comments='', fmt='%.17g')
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
