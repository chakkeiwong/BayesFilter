"""Direct Stan 2.37 Euclidean multinomial NUTS for row-independent TF targets.

Standalone research implementation; not a BayesFilter public tuner. Numerical
initialization, adaptation and sampling are enclosed by stable-signature XLA
functions. The caller owns unconstraining transforms and their Jacobians.
Random streams use TensorFlow Philox, not Stan's MIXMAX/Boost distributions.

Translated from Stan's base_nuts.hpp, base_hmc.hpp, windowed_adaptation.hpp
and Euclidean Hamiltonians. Copyright (c) 2011--2020, Stan Developers and
their Assignees. Distributed under the BSD 3-Clause license retained in
stan_nuts_LICENSE.txt. See the execution plan and native-header oracle tests.
"""
from dataclasses import dataclass
import math
from numbers import Real
from typing import NamedTuple

if __package__:
    from .stan_nuts_warmup import dual_update, require_valid, SampleResult
else:
    from stan_nuts_warmup import dual_update, require_valid, SampleResult


@dataclass(frozen=True)
class Config:
    dimension: int
    chains: int = 1
    warmup: int = 1000
    draws: int = 2000  # Number of transitions, before thinning (Stan services).
    metric: str = 'diag'
    depth: int = 10
    delta: float = .8
    initial_epsilon: float = 1.
    initial_buffer: int = 75
    terminal_buffer: int = 50
    window: int = 25
    gamma: float = .05
    kappa: float = .75
    t0: float = 10.
    adapt: bool = True
    jitter: float = 0.
    max_delta: float = 1000.
    thin: int = 1
    init_radius: float = 2.

    def __post_init__(self):
        for name in ('dimension', 'chains', 'warmup', 'draws', 'depth',
                     'initial_buffer', 'terminal_buffer', 'window', 'thin'):
            value = getattr(self, name)
            lower = 1 if name in ('chains', 'depth', 'thin') else 0
            if type(value) is not int or not lower <= value <= 2**31-1:
                raise ValueError(f'{name} must be an integer >= {lower}')
        if self.depth > 30:
            raise ValueError('depth must be <=30 for int32 work counters')
        if self.metric not in ('unit', 'diag', 'dense') or type(self.adapt) is not bool:
            raise ValueError('invalid metric/adapt option')
        for name in ('delta', 'initial_epsilon', 'gamma', 'kappa', 't0',
                     'jitter', 'max_delta', 'init_radius'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v):
                raise ValueError(f'{name} must be finite')
        if not 0 < self.delta < 1 or min(self.initial_epsilon, self.gamma, self.kappa, self.t0) <= 0:
            raise ValueError('invalid adaptation constants/step size')
        if not 0 <= self.jitter < 1 or self.max_delta <= 0 or self.init_radius < 0:
            raise ValueError('invalid jitter/energy threshold/initialization radius')


class Phase(NamedTuple):
    q: object
    p: object
    v: object
    g: object


class Node(NamedTuple):
    proposal: Phase
    begin: object
    end: object
    rho: object
    weight: object


class Transition(NamedTuple):
    phase: Phase
    seed: object
    acceptance: object
    leaps: object
    depth: object
    divergent: object
    energy: object
    epsilon: object
    valid: object


class FrozenState(NamedTuple):
    position: object
    step_size: object
    inverse_metric: object
    seed: object


class WarmupResult(NamedTuple):
    position: object
    step_size: object
    inverse_metric: object
    seed: object
    draws: object
    telemetry: object
    valid: object
    completion_token: object


def freeze(warmed):
    require_valid(warmed)
    return FrozenState(*warmed[:4])


def schedule(config):
    """Actual 2.37 window flags, including its early-return fallback behavior.

    A new native adapter starts with next_window=UINT_MAX. The short-budget
    fallback sets lengths but does not restart: it accumulates without updates.
    Zero windows/buffers follow unsigned counters, not an invented repair.
    """
    n = config.warmup
    if n < 20 or not config.adapt or config.metric == 'unit':
        return [False]*n, [False]*n
    initial, terminal, window = config.initial_buffer, config.terminal_buffer, config.window
    if initial + terminal + window > n:
        initial, terminal = int(.15*n), int(.10*n)
        return [initial <= i < n-terminal for i in range(n)], [False]*n
    end = (initial + window - 1) % 2**32
    active, ends = [], []
    for i in range(n):
        active.append(initial <= i < n-terminal)
        # learn_variance/covariance only checks window end inside active window.
        boundary = active[-1] and i == end
        ends.append(boundary)
        if boundary and end != n-terminal-1:
            window *= 2
            end = i + window
            if end != n-terminal-1 and end + 2*window >= n-terminal:
                end = n-terminal-1
    return active, ends


class Kernel:
    """One numerical authority used by scalar/batch, oracle and lifecycle paths."""
    def __init__(self, tf, log_prob, config, *, uniform_tape=None, momentum=None):
        self.tf, self.log_prob, self.config = tf, log_prob, config
        self.c, self.d = config.chains, config.dimension
        self.dtype = tf.float64
        self.tape, self.supplied_momentum = uniform_tape, momentum

    def choose(self, mask, yes, no):
        tf = self.tf
        def select(a, b):
            shape = [self.c] + [1]*(a.shape.rank-1)
            return tf.where(tf.reshape(mask, shape), a, b)
        return tf.nest.map_structure(select, yes, no)

    def valid_seed(self, seed):
        tf = self.tf
        # State is (stream key, counter), NOT two interchangeable seed words.
        # Equal keys with shifted counters would overlap random streams.
        equal = seed[:,0,None] == seed[None,:,0]
        return (tf.reduce_sum(tf.cast(equal,tf.int32),-1) == 1) & (seed[:,1] >= 0)

    def random(self, seed, mask, *, normal=False, size=()):
        tf = self.tf
        if self.tape is not None and not normal:
            value = tf.gather(self.tape, seed[:, 1], batch_dims=1)
        else:
            fn = tf.random.stateless_normal if normal else tf.random.stateless_uniform
            if self.c == 1:
                value = fn(size, seed[0], dtype=self.dtype, alg='philox')[None]
            else:
                value = tf.map_fn(lambda s: fn(size, s, dtype=self.dtype, alg='philox'),
                    seed, fn_output_signature=tf.TensorSpec(size, self.dtype),
                    parallel_iterations=1)
        return value, seed + tf.stack((tf.zeros([self.c], tf.int32), tf.cast(mask, tf.int32)), -1)

    def value_score(self, q):
        tf = self.tf
        with tf.GradientTape() as tape:
            tape.watch(q)
            lp = self.log_prob(q)
            if not tf.is_tensor(lp) or lp.dtype != self.dtype:
                raise TypeError('target must return float64 Tensor [chains]')
            lp = tf.ensure_shape(lp, [self.c])
        g = tape.gradient(lp, q)
        if g is None:
            if self.d:
                raise ValueError('target must be differentiable in position')
            g = tf.zeros_like(q)
        return -lp, -g

    def phase(self, q, p):
        v, g = self.value_score(q)
        return Phase(q, p, v, g)

    def health(self, z):
        tf = self.tf
        return (tf.reduce_all(tf.math.is_finite(z.q), -1) & tf.math.is_finite(z.v)
                & tf.reduce_all(tf.math.is_finite(z.g), -1))

    def velocity(self, p, metric):
        if self.config.metric == 'unit':
            return p
        if self.config.metric == 'diag':
            return metric*p
        return self.tf.linalg.matvec(metric, p)

    def factor(self, metric):
        if self.config.metric == 'dense':
            return self.tf.linalg.cholesky(metric)
        return self.tf.sqrt(metric)

    def momentum(self, seed, mask, factor):
        if self.supplied_momentum is not None:
            return self.supplied_momentum, seed
        tf = self.tf
        p, seed = self.random(seed, mask, normal=True, size=(self.d,))
        if self.config.metric == 'dense':
            p = tf.linalg.triangular_solve(factor, p[..., None], adjoint=True)[..., 0]
        elif self.config.metric == 'diag':
            p = p/factor
        return p, seed

    def energy(self, z, metric):
        return z.v + .5*self.tf.reduce_sum(z.p*self.velocity(z.p, metric), -1)

    def leapfrog(self, z, epsilon, metric):
        p = z.p - .5*epsilon[:, None]*z.g
        q = z.q + epsilon[:, None]*self.velocity(p, metric)
        v, g = self.value_score(q)
        p = p - .5*epsilon[:, None]*g
        return Phase(q, p, v, g)

    def criterion(self, begin, end, rho, metric):
        tf = self.tf
        return ((tf.reduce_sum(self.velocity(end, metric)*rho, -1) > 0.)
                & (tf.reduce_sum(self.velocity(begin, metric)*rho, -1) > 0.))

    def criteria(self, left, right, metric):
        return (self.criterion(left.begin, right.end, left.rho+right.rho, metric)
            & self.criterion(left.begin, right.begin, left.rho+right.begin, metric)
            & self.criterion(left.end, right.end, right.rho+left.end, metric))

    def logadd(self, a, b):
        tf = self.tf
        # Source log_sum_exp uses max + log1p(exp(min-max)); handles -inf.
        m = tf.maximum(a, b)
        return tf.where(tf.math.is_inf(m), m,
                        m + tf.math.log1p(tf.exp(tf.minimum(a, b)-m)))

    def subtree(self, z, depth, direction, epsilon, metric, h0, seed, active):
        """Depth-first leaves plus binary-carry merges: O(depth*dimension) stack.

        The carry schedule is the postorder of Stan's recursive build_tree.
        Early failure stops immediately; unsuccessful proposals never escape
        to the outer trajectory. Masking never advances a stopped chain RNG.
        """
        tf = self.tf
        zeros = tf.zeros([self.c], self.dtype)
        node = Node(z, z.p, z.p, tf.zeros_like(z.p), tf.fill([self.c], tf.constant(-math.inf, self.dtype)))
        stack = tf.nest.map_structure(lambda x: tf.zeros([self.config.depth]+x.shape.as_list(), x.dtype), node)
        def leaf_body(k, z, node, stack, seed, alive, leaps, accept, diverged):
            next_z = self.leapfrog(z, direction*epsilon, metric)
            z = self.choose(alive, next_z, z)
            h = self.energy(z, metric)
            h = tf.where(tf.math.is_nan(h), tf.constant(math.inf, self.dtype), h)
            logw = h0-h
            divergent = h-h0 > self.config.max_delta
            leaps += tf.cast(alive, tf.int32)
            accept += tf.where(alive, tf.exp(tf.minimum(logw, 0.)), 0.)
            diverged |= alive & divergent
            node = Node(z, z.p, z.p, z.p, logw)
            alive &= ~divergent
            # Merge completed left siblings, youngest first. All live chains
            # have evaluated k+1 leaves; stopped chains are masked independently.
            def merge_cond(level, carry, node, seed, alive):
                return (level < depth) & (carry % 2 == 0) & tf.reduce_any(alive)
            def merge_body(level, carry, right, seed, alive):
                left = tf.nest.map_structure(lambda x: x[level], stack)
                weight = self.logadd(left.weight, right.weight)
                automatic = right.weight > weight
                u, seed = self.random(seed, alive & ~automatic)
                take = automatic | (u < tf.exp(right.weight-weight))
                proposal = self.choose(take, right.proposal, left.proposal)
                merged = Node(proposal, left.begin, right.end, left.rho+right.rho, weight)
                alive &= self.criteria(left, right, metric)
                return level+1, carry//2, merged, seed, alive
            level, _, node, seed, alive = tf.while_loop(merge_cond, merge_body,
                (tf.constant(0), k+1, node, seed, alive), parallel_iterations=1)
            stack = tf.nest.map_structure(
                lambda a, b: tf.tensor_scatter_nd_update(a, tf.reshape(level, [1,1]), b[None]), stack, node)
            return k+1, z, node, stack, seed, alive, leaps, accept, diverged
        result = tf.while_loop(
            lambda k,z,node,stack,seed,alive,*_: (k < tf.bitwise.left_shift(1, depth)) & tf.reduce_any(alive),
            leaf_body, (tf.constant(0),z,node,stack,seed,active,tf.zeros([self.c],tf.int32),
                        zeros,tf.zeros([self.c],tf.bool)), parallel_iterations=1)
        _, z, node, _, seed, alive, leaps, accept, diverged = result
        return z, node, seed, alive, leaps, accept, diverged

    def transition(self, q, epsilon, metric, factor, seed):
        tf = self.tf
        active = tf.ones([self.c], tf.bool)
        if self.config.jitter:
            u, seed = self.random(seed, active)
            epsilon = epsilon*(1.+self.config.jitter*(2.*u-1.))
        p, seed = self.momentum(seed, active, factor)
        initial = self.phase(q, p)
        h0 = self.energy(initial, metric)
        old = Node(initial, p, p, p, tf.zeros([self.c], self.dtype))
        def body(level, backward, forward, old, seed, alive, leaps, accept, depths, divergent):
            u, seed = self.random(seed, alive)
            positive = u > .5
            endpoint = self.choose(positive, forward, backward)
            endpoint, new, seed, good, n, a, div = self.subtree(
                endpoint, level, tf.where(positive, tf.ones_like(epsilon), -tf.ones_like(epsilon)),
                epsilon, metric, h0, seed, alive)
            forward = self.choose(positive & alive, endpoint, forward)
            backward = self.choose(~positive & alive, endpoint, backward)
            leaps += n
            accept += a
            divergent |= div
            depths += tf.cast(good, tf.int32)
            automatic = new.weight > old.weight
            u, seed = self.random(seed, good & ~automatic)
            take = good & (automatic | (u < tf.exp(new.weight-old.weight)))
            proposal = self.choose(take, new.proposal, old.proposal)
            # Internal tree endpoints follow traversal order. A backward subtree
            # is reversed for the physical-time outer checks.
            reverse_new = Node(new.proposal,new.end,new.begin,new.rho,new.weight)
            left = self.choose(positive, old, reverse_new)
            right = self.choose(positive, new, old)
            persist = self.criteria(left, right, metric)
            merged = Node(proposal, left.begin, right.end, left.rho+right.rho,
                          self.logadd(old.weight,new.weight))
            old = self.choose(good, merged, old)
            return level+1,backward,forward,old,seed,good & persist,leaps,accept,depths,divergent
        result = tf.while_loop(lambda level,b,f,o,s,alive,*_: (level < self.config.depth) & tf.reduce_any(alive),
            body, (tf.constant(0),initial,initial,old,seed,active,tf.zeros([self.c],tf.int32),
                   tf.zeros([self.c],self.dtype),tf.zeros([self.c],tf.int32),tf.zeros([self.c],tf.bool)),
            parallel_iterations=1)
        _,_,_,node,seed,_,leaps,accept,depths,divergent = result
        acceptance = accept/tf.cast(leaps,self.dtype)
        valid = self.health(initial) & self.health(node.proposal) & tf.math.is_finite(acceptance) & self.valid_seed(seed)
        return Transition(node.proposal,seed,acceptance,leaps,depths,divergent,
                          self.energy(node.proposal,metric),epsilon,valid)

    def find_epsilon(self, q, epsilon, metric, factor, seed):
        """Stan's duplicated first probe and independent per-chain search state."""
        tf = self.tf
        initial = self.phase(q, tf.zeros_like(q))
        good = self.health(initial) & tf.math.is_finite(epsilon) & (epsilon > 0.)
        eligible = good & (epsilon <= 1e7)
        def probe(e, s, mask):
            p, s = self.momentum(s, mask, factor)
            z = initial._replace(p=p)
            end = self.leapfrog(z,e,metric)
            h = self.energy(end,metric)
            h = tf.where(tf.math.is_nan(h),tf.constant(math.inf,self.dtype),h)
            return self.energy(z,metric)-h,s
        first,seed = tf.cond(tf.reduce_any(eligible),lambda:probe(epsilon,seed,eligible),
                            lambda:(tf.zeros_like(epsilon),seed))
        threshold = tf.math.log(tf.constant(.8,self.dtype))
        direction = first > threshold
        def body(i,e,s,pending,count,good):
            loga,s = probe(e,s,pending)
            go = pending & tf.where(direction,loga > threshold,loga < threshold)
            e = tf.where(go,e*tf.where(direction,tf.constant(2.,self.dtype),tf.constant(.5,self.dtype)),e)
            bounded = (e > 0.) & (e <= 1e7) & tf.math.is_finite(e)
            return i+1,e,s,go & bounded,count+tf.cast(pending,tf.int32),good & ~(go & ~bounded)
        _,e,seed,pending,count,good = tf.while_loop(
            lambda i,e,s,p,*_: (i < 2048) & tf.reduce_any(p), body,
            (tf.constant(0),epsilon,seed,eligible,tf.cast(eligible,tf.int32),good),parallel_iterations=1)
        return e,seed,count,good & ~pending

    def valid_metric(self, metric, factor):
        tf = self.tf
        axes = [-2,-1] if self.config.metric == 'dense' else [-1]
        ok = tf.reduce_all(tf.math.is_finite(metric),axes) & tf.reduce_all(tf.math.is_finite(factor),axes)
        if self.config.metric == 'dense':
            # Stan Math check_symmetric uses absolute CONSTRAINT_TOLERANCE=1e-8.
            ok &= tf.reduce_all(tf.abs(metric-tf.linalg.matrix_transpose(metric)) <= 1e-8,[-2,-1])
            ok &= tf.reduce_all(tf.linalg.diag_part(factor) > 0.,-1)
        else:
            ok &= tf.reduce_all(metric > 0.,-1)
            if self.config.metric == 'unit':
                ok &= tf.reduce_all(metric == 1.,-1)
        return ok

    def telemetry(self, result, previous, valid):
        tf = self.tf
        return tf.stack((result.acceptance,
            tf.cast(tf.reduce_any(result.phase.q != previous,-1),self.dtype),
            tf.cast(result.leaps,self.dtype),tf.cast(result.depth >= self.config.depth,self.dtype),
            tf.cast(result.divergent,self.dtype),tf.cast(valid,self.dtype),
            tf.cast(result.depth,self.dtype),result.energy,result.epsilon),-1)


def make_functions(tf, log_prob, config, *, model_signature=(), kernel_type=None):
    """Return compiled initialize, warmup, sample functions.

    Seeds are [chains,2] int32 (distinct stream key, nonnegative counter) states.
    Do not initialize chains by offsetting counters under one shared key.
    Sampling returns the exact continuation
    state. Metrics are inverse masses: [chains,d] or [chains,d,d]. Both warmup
    and sampling retain zero-based iterations 0,thin,2*thin,...; warmup draws
    are separate and must not be included in posterior summaries.
    Optional model_signature contains fully defined numeric/boolean TensorSpecs;
    matching tensor arguments are appended to each function and passed to
    log_prob(q, *model_inputs). kernel_type is an internal implementation hook
    for controlled graph-layout comparisons, not an alternate adaptation.
    """
    # Model inputs stay tensor operands in the enclosing compiled computation.
    # A factory extension avoids duplicating the source-audited lifecycle.
    model_signature = tuple(model_signature)
    for spec in model_signature:
        if not isinstance(spec, tf.TensorSpec) or not spec.shape.is_fully_defined():
            raise ValueError('model_signature requires fully defined TensorSpecs')
        if not (spec.dtype.is_floating or spec.dtype.is_integer or spec.dtype == tf.bool):
            raise TypeError('model inputs must be numeric or boolean tensors')
    kernel_type = Kernel if kernel_type is None else kernel_type
    def bind(model_inputs):
        return kernel_type(tf, lambda q: log_prob(q, *model_inputs), config)
    c,d = config.chains,config.dimension
    dt = tf.float64
    shape = [c,d,d] if config.metric == 'dense' else [c,d]
    qs,rs = tf.TensorSpec([c,d],dt),tf.TensorSpec([c,2],tf.int32)
    ms,es = tf.TensorSpec(shape,dt),tf.TensorSpec([c],dt)

    @tf.function(input_signature=[qs,tf.TensorSpec([c,d],tf.bool),rs,*model_signature],autograph=False,jit_compile=True)
    def initialize(supplied, present, seed, *model_inputs):
        k = bind(model_inputs)
        full = tf.reduce_all(present,-1)
        attempts = tf.where(full | (config.init_radius == 0.),1,100)
        def body(i,q,seed,good,counts):
            pending = ~good & (counts < attempts)
            if config.init_radius:
                u,seed = k.random(seed,pending,size=(d,))
                candidate = (2.*u-1.)*config.init_radius
            else:
                candidate = tf.zeros_like(q)
            candidate = tf.where(present,supplied,candidate)
            z = k.phase(candidate,tf.zeros_like(q))
            # initialize.hpp checks finite sum of gradients (not just elements).
            ok = k.health(z) & tf.math.is_finite(tf.reduce_sum(z.g,-1))
            q = k.choose(pending,candidate,q)
            return i+1,q,seed,good | (pending & ok),counts+tf.cast(pending,tf.int32)
        _,q,seed,good,counts = tf.while_loop(
            lambda i,q,s,g,n: tf.reduce_any(~g & (n < attempts)),body,
            (tf.constant(0),supplied,seed,tf.zeros([c],tf.bool),tf.zeros([c],tf.int32)),parallel_iterations=1)
        return q,seed,good & k.valid_seed(seed),counts

    active,ends = schedule(config)

    @tf.function(input_signature=[qs,es,ms,rs,*model_signature],autograph=False,jit_compile=True)
    def warmup(q,e,metric,seed,*model_inputs):
        k = bind(model_inputs)
        factor = k.factor(metric)
        good = k.health(k.phase(q,tf.zeros_like(q))) & k.valid_metric(metric,factor) & k.valid_seed(seed)
        good &= tf.math.is_finite(e) & (e > 0.)
        mu = tf.math.log(10.*e)
        probes = tf.zeros([c],tf.int32)
        if config.adapt and d:
            e,seed,probes,ok = k.find_epsilon(q,e,metric,factor,seed)
            good &= ok
        if config.warmup == 0:
            return WarmupResult(q,e,metric,seed,tf.zeros([0,c,d],dt),tf.zeros([0,c,13],dt),
                                good,tf.reduce_sum(q)+tf.reduce_sum(e))
        nsave = (config.warmup+config.thin-1)//config.thin
        saved = tf.TensorArray(dt,size=nsave,element_shape=[c,d])
        trace = tf.TensorArray(dt,size=nsave,element_shape=[c,13])
        def body(i,q,e,metric,factor,seed,n,h,x,mu,wn,mean,m2,probes,good,saved,trace):
            previous = q
            if d:
                tr = k.transition(q,e,metric,factor,seed)
                q,seed = tr.phase.q,tr.seed
                good &= tr.valid
                row = k.telemetry(tr,previous,good)
            else:
                row = tf.stack((tf.ones([c],dt),tf.zeros([c],dt),tf.zeros([c],dt),
                    tf.zeros([c],dt),tf.zeros([c],dt),tf.cast(good,dt),tf.zeros([c],dt),
                    k.phase(q,q).v,tf.fill([c],tf.constant(math.nan,dt))),-1)
            boundary = tf.constant(ends,tf.bool)[i]
            if config.adapt and d:
                n,h,x,e = dual_update(tf,n,h,x,mu,row[:,0],config)
                def update():
                    new_n = wn+1
                    diff = q-mean
                    new_mean = mean+diff/tf.cast(new_n,dt)
                    update = ((q-new_mean)[:,:,None]*diff[:,None,:]
                              if config.metric == 'dense' else (q-new_mean)*diff)
                    return new_n,new_mean,m2+update
                wn,mean,m2 = tf.cond(tf.constant(active,tf.bool)[i],update,lambda:(wn,mean,m2))
                def change_metric():
                    count = tf.cast(wn,dt)
                    cov = tf.where(wn > 1,m2/tf.maximum(count-1.,1.),metric)
                    identity = tf.eye(d,batch_shape=[c],dtype=dt) if config.metric == 'dense' else tf.ones([c,d],dt)
                    new_metric = count/(count+5.)*cov+1e-3*5./(count+5.)*identity
                    new_factor = k.factor(new_metric)
                    new_e,new_seed,count_probe,ok = k.find_epsilon(q,e,new_metric,new_factor,seed)
                    return (new_e,new_metric,new_factor,new_seed,tf.constant(0),tf.zeros_like(h),
                        tf.zeros_like(x),tf.math.log(10.*new_e),tf.constant(0),tf.zeros_like(mean),
                        tf.zeros_like(m2),probes+count_probe,good & ok & k.valid_metric(new_metric,new_factor))
                e,metric,factor,seed,n,h,x,mu,wn,mean,m2,probes,good = tf.cond(boundary,change_metric,
                    lambda:(e,metric,factor,seed,n,h,x,mu,wn,mean,m2,probes,good))
                good &= tf.math.is_finite(e) & (e > 0.)
            detail = tf.stack((e,tf.fill([c],tf.cast(boundary,dt)),tf.cast(probes,dt),tf.cast(good,dt)),-1)
            saved,trace = tf.cond(i % config.thin == 0,
                lambda:(saved.write(i//config.thin,q),trace.write(i//config.thin,tf.concat((row,detail),-1))),
                lambda:(saved,trace))
            return i+1,q,e,metric,factor,seed,n,h,x,mu,wn,mean,m2,probes,good,saved,trace
        out = tf.while_loop(lambda i,*_: i < config.warmup,body,
            (tf.constant(0),q,e,metric,factor,seed,tf.constant(0),tf.zeros([c],dt),tf.zeros([c],dt),mu,
             tf.constant(0),tf.zeros([c,d],dt),tf.zeros(shape,dt),probes,good,saved,trace),parallel_iterations=1)
        _,q,e,metric,factor,seed,n,h,x,mu,wn,mean,m2,probes,good,saved,trace = out
        if config.adapt and d:
            e = tf.where(n > 0,tf.exp(x),e)
            good &= tf.math.is_finite(e) & (e > 0.)
        draws,telemetry = saved.stack(),trace.stack()
        token = tf.reduce_sum(q)+tf.reduce_sum(e)+tf.reduce_sum(metric)
        return WarmupResult(q,e,metric,seed,draws,telemetry,good,token)

    @tf.function(input_signature=[qs,es,ms,rs,*model_signature],autograph=False,jit_compile=True)
    def sample(q,e,metric,seed,*model_inputs):
        k = bind(model_inputs)
        factor = k.factor(metric)
        good = k.health(k.phase(q,tf.zeros_like(q))) & k.valid_metric(metric,factor) & k.valid_seed(seed)
        good &= tf.math.is_finite(e) & (e > 0.)
        if config.draws == 0:
            return SampleResult(q,seed,tf.zeros([0,c,d],dt),tf.zeros([0,c,9],dt),good,tf.reduce_sum(q))
        nsave = (config.draws+config.thin-1)//config.thin
        saved = tf.TensorArray(dt,size=nsave,element_shape=[c,d])
        trace = tf.TensorArray(dt,size=nsave,element_shape=[c,9])
        def body(i,q,seed,good,saved,trace):
            previous = q
            if d:
                tr = k.transition(q,e,metric,factor,seed)
                q,seed = tr.phase.q,tr.seed
                good &= tr.valid
                row = k.telemetry(tr,previous,good)
            else:
                row = tf.stack((tf.ones([c],dt),tf.zeros([c],dt),tf.zeros([c],dt),tf.zeros([c],dt),
                    tf.zeros([c],dt),tf.cast(good,dt),tf.zeros([c],dt),k.phase(q,q).v,
                    tf.fill([c],tf.constant(math.nan,dt))),-1)
            saved,trace = tf.cond(i % config.thin == 0,
                lambda:(saved.write(i//config.thin,q),trace.write(i//config.thin,row)),lambda:(saved,trace))
            return i+1,q,seed,good,saved,trace
        _,q,seed,good,saved,trace = tf.while_loop(lambda i,*_: i < config.draws,body,
            (tf.constant(0),q,seed,good,saved,trace),parallel_iterations=1)
        draws,telemetry = saved.stack(),trace.stack()
        return SampleResult(q,seed,draws,telemetry,good,tf.reduce_sum(q)+tf.cast(tf.reduce_sum(seed),dt))
    return initialize,warmup,sample
