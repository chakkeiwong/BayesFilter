"""Independent CPU/NumPy diagnostic of the author's LAPACK arithmetic.

Not a TensorFlow dependency or a candidate runtime. The library and saved
inputs are checked against the preceding source-reference fixture.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from reference_octave_system_lapack import SystemLapack


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditLapack(SystemLapack):
    def scalar(self, name, text):
        function = getattr(self.library, name + '_')
        function.restype = ct.c_double
        return float(function(text.encode(), ct.c_size_t(len(text))))

    def ilaenv(self, spec, name, options, dims):
        function = self.library.ilaenv_
        function.restype = ct.c_int
        values = [ct.c_int(x) for x in (spec, *dims)]
        return int(function(ct.byref(values[0]), name.encode(), options.encode(),
                            *(ct.byref(x) for x in values[1:]),
                            ct.c_size_t(len(name)), ct.c_size_t(len(options))))

    def least_squares(self, name, g, h):
        n = len(h); ni = ct.c_int(n); one = ct.c_int(1)
        a = np.array(g, order='F'); b = h.copy(); spectrum = np.zeros(n)
        cutoff = ct.c_double(-1.); rank = ct.c_int(); info = ct.c_int()
        query = np.zeros(1); minus = ct.c_int(-1); iw = np.zeros(100*n, np.int32)
        def call(work, length):
            args = [ni, ni, one, a, ni, b, ni, spectrum, cutoff, rank, work, length]
            if name == 'dgelsd':
                args.append(iw)
            self._call(name, *args, info)
            assert info.value == 0, (name, info.value)
        call(query, minus)
        query_unchanged = np.array_equal(a, g) and np.array_equal(b, h)
        assert query_unchanged
        length = ct.c_int(int(query[0])); work = np.zeros(length.value)
        call(work, length)
        return b, spectrum, dict(rank=rank.value, lwork=length.value,
                                 workspace_query_preserves_inputs=query_unchanged)

    def explicit_svd(self, name, g, h, eps):
        n = len(h); ni = ct.c_int(n); info = ct.c_int()
        a = np.array(g, order='F'); spectrum = np.zeros(n)
        u = np.zeros((n, n), order='F'); vt = np.zeros((n, n), order='F')
        query = np.zeros(1); iw = np.zeros(8*n, np.int32)
        def call(work, length):
            if name == 'dgesvd':
                self._call(name, b'A', b'A', ni, ni, a, ni, spectrum, u, ni, vt,
                           ni, work, length, info, ct.c_size_t(1), ct.c_size_t(1))
            else:
                self._call(name, b'A', ni, ni, a, ni, spectrum, u, ni, vt, ni,
                           work, length, iw, info, ct.c_size_t(1))
            assert info.value == 0, (name, info.value)
        call(query, ct.c_int(-1))
        assert np.array_equal(a, g)
        call(np.zeros(int(query[0])), ct.c_int(int(query[0])))
        retained = spectrum > eps * spectrum[0]
        inv = np.zeros(n); inv[retained] = 1 / spectrum[retained]
        beta = vt.T @ (inv * (u.T @ h))
        return beta, spectrum, dict(rank=int(retained.sum()),
            reconstruction_max_abs=float(np.max(np.abs((u*spectrum)@vt-g))))

    def bidiagonal_chain(self, g, h, lwork, smlsiz, mode='source'):
        n = len(h); ni = ct.c_int(n); one = ct.c_int(1); info = ct.c_int()
        a = np.array(g, order='F'); b = h.copy()
        d = np.zeros(n); e = np.zeros(n); tq = np.zeros(n); tp = np.zeros(n)
        # Same lengths and calls as DGELSD Path 1 without preliminary QR.
        work = np.zeros(lwork - 3*n); length = ct.c_int(len(work))
        self._call('dgebrd', ni, ni, a, ni, d, e, tq, tp, work, length, info)
        assert info.value == 0
        self._call('dormbr', b'Q', b'L', b'T', ni, one, ni, a, ni, tq, b, ni,
                   work, length, info, ct.c_size_t(1), ct.c_size_t(1), ct.c_size_t(1))
        assert info.value == 0
        raw_d = d.copy(); raw_e = e.copy(); projected_h = b.copy()
        eps = self.scalar('dlamch', 'E')
        norm = max(np.max(np.abs(d)), np.max(np.abs(e[:-1])))
        floor = np.flatnonzero(np.abs(d/norm) < eps)
        split = np.flatnonzero(np.abs(e[:-1]/norm) < eps)
        if mode == 'source' or mode == 'small_problem_qr':
            if mode == 'small_problem_qr':
                smlsiz = n
            # Work is oversized only for this diagnostic branch; minimum
            # storage formulas are documented by DLALSD. No solver cap changes.
            bw = np.zeros(max(len(work), 20*n*n)); iw = np.zeros(100*n, np.int32)
            rank = ct.c_int()
            self._call('dlalsd', b'U', ct.c_int(smlsiz), ni, one, d, e, b, ni,
                       ct.c_double(-1.), rank, bw, iw, info, ct.c_size_t(1))
            assert info.value == 0
            rank_value = rank.value
        else:
            # Apply source preprocessing interventions, then use the same
            # DLASDQ QR solver. The controls separate flooring and splitting.
            zero = ct.c_int(0)
            self._call('dlascl', b'G', zero, zero, ct.c_double(norm), ct.c_double(1.),
                       ni, one, d, ni, info, ct.c_size_t(1))
            assert info.value == 0
            self._call('dlascl', b'G', zero, zero, ct.c_double(norm), ct.c_double(1.),
                       ct.c_int(n-1), one, e, ct.c_int(n-1), info, ct.c_size_t(1))
            assert info.value == 0
            if mode in ('qr_floor', 'qr_floor_split'):
                d[floor] = np.copysign(eps, d[floor])
            if mode in ('qr_split', 'qr_floor_split'):
                e[split] = 0.
            vt = np.eye(n, order='F'); unused_u = np.zeros(1)
            self._call('dlasdq', b'U', zero, ni, ni, zero, one, d, e, vt, ni,
                       unused_u, one, b, ni, np.zeros(4*n), info, ct.c_size_t(1))
            assert info.value == 0
            retained = d > eps*np.max(d); rank_value = int(retained.sum())
            b[~retained] = 0.
            # DLASCL also follows the source row scaling, avoiding reciprocal
            # multiplication as an accidental change in this mechanism test.
            for i in np.flatnonzero(retained):
                self._call('dlascl', b'G', zero, zero, ct.c_double(d[i]), ct.c_double(1.),
                           one, one, b[i:i+1], one, info, ct.c_size_t(1))
                assert info.value == 0
            b[:] = vt.T @ b
            self._call('dlascl', b'G', zero, zero, ct.c_double(norm), ct.c_double(1.),
                       ni, one, b, ni, info, ct.c_size_t(1))
            assert info.value == 0
            d = np.sort(np.abs(d)*norm)[::-1]
        self._call('dormbr', b'P', b'L', b'N', ni, one, ni, a, ni, tp, b, ni,
                   work, length, info, ct.c_size_t(1), ct.c_size_t(1), ct.c_size_t(1))
        assert info.value == 0
        return b, d, dict(rank=rank_value, floor_indices=floor.tolist(),
            split_after_indices=split.tolist(), bidiagonal_max_abs=float(norm),
            minimum_scaled_diagonal=float(np.min(np.abs(raw_d/norm)))), {
            'bidiagonal_d':raw_d, 'bidiagonal_e':raw_e, 'projected_rhs':projected_h}


def main(args):
    start = time.monotonic(); root = Path.cwd()
    base = root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    fixture_dir = base/'solver-compatibility-reference-20261006-01'
    fixture = json.loads((fixture_dir/'fixture.json').read_text())
    inputs = {str(fixture_dir/'fixture.json'):sha(fixture_dir/'fixture.json')}
    arrays = {}
    for key in ('gram', 'rhs', 'beta', 'design', 'target'):
        item = fixture['arrays'][key]; path = fixture_dir/item['file']
        assert sha(path) == item['sha256']
        inputs[str(path)] = sha(path)
        arrays[key] = np.fromfile(path, dtype='<f8').reshape(item['shape'])
    library = Path('/usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3')
    assert sha(library) == fixture['library_sha256']
    solver = AuditLapack(library)
    version = [ct.c_int() for _ in range(3)]; solver._call('ilaver', *version)
    assert [v.value for v in version] == [3,10,0]
    n = 45; eps = solver.scalar('dlamch','E'); precision = solver.scalar('dlamch','P')
    smlnum = solver.scalar('dlamch','S') / precision
    smlsiz = solver.ilaenv(9, 'DGELSD', ' ', (0,0,0,0))
    mnthr = solver.ilaenv(6, 'DGELSD', ' ', (n,n,1,-1))
    assert mnthr > n  # no preliminary QR on these square systems
    sources = {}
    for name in ('dgelsd','dgelss','dlalsd','dlasda','dlasdq'):
        path = args.output/(name+'.f'); shutil.copyfile('/tmp/'+name+'.f', path)
        sources[str(path)] = dict(sha256=sha(path),
            url='https://raw.githubusercontent.com/Reference-LAPACK/lapack/v3.10.0/SRC/'+name+'.f')
    cases = []
    for t in (20,25,28):
        index = next(i for i,r in enumerate(fixture['rows']) if r['label']==f'offset_time{t}')
        cases.append((f'time{t}', {k:v[index] for k,v in arrays.items()}))
    # No RNG: a dense SPD matrix and known vector, reproducible independently.
    v = np.arange(1,n+1,dtype=np.float64)/n
    control_g = np.diag(np.linspace(1.,3.,n)) + np.outer(v,v)
    control_beta = np.linspace(-1.,1.,n)
    cases.insert(0, ('known_control', dict(gram=control_g, rhs=control_g@control_beta,
        beta=control_beta, design=np.eye(n), target=control_beta)))
    saved = {}; rows = []
    for label,case in cases:
        g,h,reference,design,target = (case[k] for k in ('gram','rhs','beta','design','target'))
        assert smlnum < np.max(np.abs(g)) < 1/smlnum
        assert smlnum < np.max(np.abs(h)) < 1/smlnum
        exact_beta, exact_s, exact_meta = solver.least_squares('dgelsd', g, h)
        if label != 'known_control':
            assert np.array_equal(exact_beta, reference), label
        candidates = {'dgelsd':(exact_beta,exact_s,exact_meta)}
        candidates['dgelss'] = solver.least_squares('dgelss',g,h)
        for name in ('dgesvd','dgesdd'):
            candidates[name] = solver.explicit_svd(name,g,h,eps)
        for mode in ('source','small_problem_qr','qr_unmodified','qr_floor','qr_split','qr_floor_split'):
            beta,spectrum,meta,internal = solver.bidiagonal_chain(g,h,exact_meta['lwork'],smlsiz,mode)
            candidates['chain_'+mode] = (beta,spectrum,meta)
            if mode=='source':
                comparison = dict(case=label, coefficient_max_abs=float(np.max(np.abs(beta-exact_beta))), spectrum_max_abs=float(np.max(np.abs(spectrum-exact_s))), source_meta=exact_meta, chain_meta=meta)
                print(json.dumps(comparison), flush=True)
                if label != 'known_control':
                    np.savez(args.output/(label+'_chain_check.npz'), beta=beta, source_beta=exact_beta, spectrum=spectrum, source_spectrum=exact_s, **internal)
                    assert np.array_equal(spectrum,exact_s), label
                    reconstruction_difference = np.abs(design@beta-design@exact_beta)
                    assert np.all(reconstruction_difference <= 1e-7+1e-9*np.abs(design@exact_beta)), label
                for key,value in internal.items():
                    saved[label+'_'+key] = value
        case_rows = []
        for name,(beta,spectrum,meta) in candidates.items():
            assert np.all(np.isfinite(beta)) and np.all(np.isfinite(spectrum))
            coefficient_error = float(np.max(np.abs(beta-reference)))
            if label == 'known_control':
                assert coefficient_error < 1e-10, (name,coefficient_error)
            saved[label+'_'+name+'_beta'] = beta
            saved[label+'_'+name+'_spectrum'] = spectrum
            prediction = design @ beta
            row = dict(case=label, arm=name, exact_source=np.array_equal(beta,exact_beta),
                coefficient_error=coefficient_error,
                prediction_error=float(np.max(np.abs(prediction-design@reference))),
                residual_rms=float(np.sqrt(np.mean((prediction-target)**2))),
                minimum_singular_ratio=float(spectrum[-1]/spectrum[0]),
                backward_error=float(np.linalg.norm(g@beta-h,np.inf)/
                    (np.linalg.norm(g,np.inf)*np.linalg.norm(beta,np.inf)+np.linalg.norm(h,np.inf))),
                **meta)
            case_rows.append(row); rows.append(row)
        with (args.output/'rows.jsonl').open('a') as f:
            for row in case_rows:
                f.write(json.dumps(row, allow_nan=False)+'\n')
    np.savez(args.output/'arrays.npz', **saved)
    chain_exact = all(r['exact_source'] for r in rows if r['arm']=='chain_source')
    summary = dict(status='PASS', scientific_verdict='SOURCE_CALL_CHAIN_NUMERICALLY_REPRODUCED_NOT_BITWISE' if not chain_exact else 'SOURCE_CALL_CHAIN_REPRODUCED',
        exact_source_call_chain=chain_exact,
        diagnostic_only=True, replication='INCOMPLETE', cpu_only=True,
        gpu_intentionally_hidden=True, rows=rows, lapack_version=[v.value for v in version],
        smlsiz=smlsiz, mnthr=mnthr, source_rank_cutoff=eps, outer_scaling_applied=False,
        outer_scaling_range=[smlnum,1/smlnum], source_files=sources)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    manifest = dict(status='PASS', command=[sys.executable,*sys.argv],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=sys.executable,cpu_only=True,gpu_intentionally_hidden=True,
        plan_file='docs/plans/controlled-smc-dgelsd-arithmetic-20261006.md',
        result_file='docs/plans/controlled-smc-dgelsd-arithmetic-result-20261006.md',
        random_seeds=[20260930],randomness='none added; preserved author fixture seed',
        threads=1,wall_seconds=time.monotonic()-start,input_sha256=inputs,
        library_path=str(library),library_sha256=sha(library),source_files=sources,
        implementation_sha256={str(Path(__file__)):sha(__file__),
            str(Path(__file__).with_name('reference_octave_system_lapack.py')):
            sha(Path(__file__).with_name('reference_octave_system_lapack.py'))},
        output_sha256={str(p):sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(cases),arms=10,
        exact_source_call_chain=chain_exact,smlsiz=smlsiz,mnthr=mnthr,wall_seconds=manifest['wall_seconds'])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args();arguments.output.mkdir(parents=True,exist_ok=False)
    try:
        main(arguments)
    except BaseException:
        (arguments.output/'error.log').write_text(traceback.format_exc())
        raise
