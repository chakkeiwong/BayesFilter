"""Frozen full80 application glue; numerical authority stays in pinned sources.

Archive decoding is host I/O. No training, NumPy runtime, filter autodiff,
or alternative likelihood is introduced. See the October 30 comparison plan.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

MACRO = Path('/home/ubuntu/python/MacroFinance')
BINDING = MACRO/'docs/plans/artifacts/dz5-full80-hmc-20261021/binding.json'
BASELINE = MACRO/'docs/plans/artifacts/dz5-full80-hmc-20261021/sample-repair-20261021T115311006662Z'
EXPECTED_SCOPE = '4bd4f8f4c1f615478f00619cb566eebe9dab626761ba22c4514ba40ae39a1e6c'
EPSILON = 0.33635856610148585

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_npy(archive, name):
    """Read only bounded, C-order primitive NPY arrays; never unpickle."""
    with archive.open(name+'.npy') as f:
        magic = f.read(8)
        if magic[:6] != b'\x93NUMPY' or magic[6:8] not in (b'\x01\x00', b'\x02\x00'):
            raise ValueError('unsupported NPY header')
        n = 2 if magic[6] == 1 else 4
        size = struct.unpack('<H' if n == 2 else '<I', f.read(n))[0]
        if size > 65536:
            raise ValueError('oversized NPY header')
        header = ast.literal_eval(f.read(size).decode('latin1'))
        shape, dtype = header['shape'], header['descr']
        if header['fortran_order'] or dtype not in ('<f8', '<f4', '<i8', '<i4', '|b1'):
            raise ValueError('unsupported array storage')
        if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid array shape')
        size = math.prod(shape)*{'<f8':8,'<f4':4,'<i8':8,'<i4':4,'|b1':1}[dtype]
        if size > 256*1024*1024:
            raise ValueError('array exceeds application I/O bound')
        data = f.read(size+1)
        if len(data) != size:
            raise ValueError('array length mismatch')
        return shape, dtype, data

def load_arrays(tf, path, names):
    dtypes = {'<f8':tf.float64,'<f4':tf.float32,'<i8':tf.int64,'<i4':tf.int32,'|b1':tf.uint8}
    with zipfile.ZipFile(path) as archive:
        result = {}
        for name in names:
            shape, dtype, raw = read_npy(archive, name)
            value = tf.reshape(tf.io.decode_raw(raw, dtypes[dtype], little_endian=True), shape)
            result[name] = tf.cast(value, tf.bool) if dtype == '|b1' else value
    return result

def verify_binding():
    binding = json.loads(BINDING.read_text())
    for key in ('checkpoint','spec','training_binding','panel','confirmation'):
        if sha(binding[key]) != binding[key+'_sha256']:
            raise ValueError('changed bound input: '+key)
    drift = [p for p,h in binding['sources'].items() if sha(MACRO/p) != h]
    if set(drift) - {'scripts/dz5_successor_master.py'}:
        raise ValueError('bound source changed: '+str(drift))
    # Static local import closure includes function-local imports conservatively.
    pending = ['dz5_full80_hmc_adapter','dz5_financial_coordinates_tf','dz5_financial_target_tf']
    seen = {}
    while pending:
        module = pending.pop()
        p = MACRO/(module.replace('.', '/')+'.py')
        if not p.is_file() or module in seen:
            continue
        seen[module] = sha(p)
        tree = ast.parse(p.read_text())
        # The old factory is replaced ONLY at the host setup boundary below.
        # Its unused function-local imports lead into campaign orchestration.
        if module == 'dz5_full80_hmc_adapter':
            tree.body = [n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name == 'create_adapter')]
        for node in ast.walk(tree):
            names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name == 'filters' or name.startswith(('filters.','inference.hmc','inference.mass_matrix','inference.posterior_adapter')):
                    raise ValueError('forbidden local runtime import: '+name)
                pending.append(name)
    if 'scripts.dz5_successor_master' in seen:
        raise ValueError('drifted master is an executable dependency')
    return binding, dict(binding_sha256=sha(BINDING), nonexecuted_source_drift=drift, local_import_closure=seen)

def activate_pinned_runtime(binding):
    if 'bayesfilter' in sys.modules:
        raise RuntimeError('pinned BayesFilter must be selected before package import')
    sys.path[:0] = [str(MACRO), binding['runtime']]

def verify_loaded_runtime(binding):
    runtime = Path(binding['runtime']).resolve()
    expected = json.loads((runtime/'snapshot.json').read_text())['source_files']
    checked = {}
    for name,module in list(sys.modules.items()):
        if not (name == 'bayesfilter' or name.startswith('bayesfilter.')):
            continue
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        p = Path(filename).resolve()
        if not p.is_relative_to(runtime):
            raise ValueError('mixed BayesFilter runtime: '+str(p))
        relative = str(p.relative_to(runtime))
        if relative not in expected or sha(p) != expected[relative]:
            raise ValueError('unverified pinned source: '+relative)
        checked[relative] = expected[relative]
    if any(n in sys.modules for n in ('scripts.dz5_successor_master','dz5_financial_neutra_tf')):
        raise ValueError('unused legacy setup/campaign module imported')
    return checked

def normal_quadrature(tf, order):
    """Normalized Gaussian quadrature via symmetric Jacobi eigensystem."""
    off = tf.sqrt(tf.range(1, order, dtype=tf.float64))
    matrix = tf.linalg.diag(off, k=1)+tf.linalg.diag(off, k=-1)
    nodes, vectors = tf.linalg.eigh(matrix)
    return nodes, tf.square(vectors[0])

def create_adapter(tf, binding):
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from dz5_analytical_pricing_tf import AnalyticalPricingConfig
    from dz5_financial_contract import NAMES, PARAMETERS
    from dz5_financial_target_tf import make_financial_posterior_tf
    from dz5_financial_coordinates_tf import SupportFinancialTarget
    from dz5_full80_hmc_adapter import restore_map, FrozenComponentMap, Full80HMCAdapter
    data = load_arrays(tf, binding['panel'], ('raw_parameters','observations','log_spot'))
    with zipfile.ZipFile(binding['panel']) as archive:
        raw_data = b''.join(read_npy(archive, n)[2] for n in ('observations','log_spot'))
    model, progress = restore_map(binding)
    nodes, weights = normal_quadrature(tf, 64)
    enodes, eweights = normal_quadrature(tf, 32)
    config = AnalyticalPricingConfig('shadow_forward',1,256,1,bond_hermite_terms=None,bond_engine='integrated')
    pricing = dict(analytical_config=config, analytical_nodes=nodes, analytical_weights=weights,
                   analytical_equity_nodes=enodes, analytical_equity_weights=eweights)
    identity = dict(data=hashlib.sha256(raw_data).hexdigest(), parameters=NAMES,
                    prior=[(p.prior_location,p.prior_scale,p.transform) for p in PARAMETERS],
                    pricing=vars(config), cir_engine='prepared',pricing_profile='equity_h256_gh32_bond_gh64_v2')
    scope = 'financial_v2_neutra_'+hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    class Base:
        parameter_dim = 80
        batch = 4
        def value_score_capability(self):
            return ValueScoreCapability(value_score_authority='graph_native',xla_hmc_ready=False,
                full_chain_xla_diagnostic_ready=False,runtime_backend='tensorflow_bayesfilter',
                target_scope=self.scope,nonclaims=('qualification required',))
    base = Base()
    base.scope = scope
    base.location = tf.constant([p.prior_location for p in PARAMETERS],tf.float64)
    base.scale = tf.constant([p.prior_scale for p in PARAMETERS],tf.float64)
    base.truth = data['raw_parameters'][0]
    base.posterior_data = tuple(tf.repeat(data[n],4,axis=0) for n in ('observations','log_spot'))
    base.posterior_pricing = dict(prepare_fixed_cir_points=True, **pricing)
    evaluate = make_financial_posterior_tf(*base.posterior_data, fallback_raw=base.truth, **base.posterior_pricing)
    def transformed(x):
        report = evaluate(base.location+base.scale*x)
        return {**report,'log_prob':report['log_prob']+tf.reduce_sum(tf.math.log(base.scale)),
                'score':report['score']*base.scale}
    signature = [tf.TensorSpec([4,80],tf.float64)]
    base.ordinary = tf.function(transformed,input_signature=signature,autograph=False)
    base.compiled = tf.function(transformed,input_signature=signature,autograph=False,jit_compile=True)
    support = SupportFinancialTarget(base)
    frozen = FrozenComponentMap(model,dict(checkpoint_sha256=binding['checkpoint_sha256'],
        spec_sha256=binding['spec_sha256'],sources=binding['sources']),batch=4)
    adapter = Full80HMCAdapter(support,frozen)
    if adapter.scope != EXPECTED_SCOPE:
        raise ValueError('original transformed target scope mismatch: '+adapter.scope)
    return adapter, progress

def score_model(tf, evaluate):
    """First-score bridge; never differentiate the physical filter."""
    @tf.custom_gradient
    def model(q):
        value, score, status = evaluate(q)
        valid = (status['status_code'] == 0) & status['valid_pre_regularized_score']
        value = tf.where(valid,value,tf.constant(float('nan'),tf.float64))
        score = tf.where(valid[:,None],score,tf.constant(float('nan'),tf.float64))
        value, score = tf.stop_gradient(value), tf.stop_gradient(score)
        def gradient(dy, variables=None):
            result = dy[:,None]*score
            return result if variables is None else (result,[None for _ in variables])
        return value, gradient
    return model

def check_state(tf, state, evaluate):
    tf.debugging.assert_equal(state.valid,tf.ones_like(state.valid))
    tf.debugging.assert_equal(state.inverse_mass,tf.ones_like(state.inverse_mass))
    tf.debugging.assert_all_finite(state.q,'retained position')
    lp, score, status = evaluate(state.q)
    tf.debugging.assert_equal(status['status_code'],tf.zeros_like(status['status_code']))
    tf.debugging.assert_equal(status['valid_pre_regularized_score'],tf.ones_like(state.valid))
    tf.debugging.assert_near(state.lp,lp,atol=2e-6,rtol=2e-7)
    tf.debugging.assert_near(state.g,score,atol=2e-6,rtol=2e-7)
