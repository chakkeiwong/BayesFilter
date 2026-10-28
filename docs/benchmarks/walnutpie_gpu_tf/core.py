"""Batch-native port of pinned Walnutpie 5710b033 (MIT; see LICENSE).

Log density/gradient signs follow upstream. All numerical loops are TensorFlow
while loops. Python loops only construct bounded tensor structures at trace time.
No CPU callback, pfor, NumPy, or Stan tree/adaptation implementation is used.
"""
from dataclasses import dataclass
from typing import NamedTuple
import math


@dataclass(frozen=True)
class Config:
    chains: int
    dimension: int
    warmup: int = 500
    draws: int = 500
    max_depth: int = 5
    max_halvings: int = 5
    min_micro_steps: int = 1
    max_error: float = .5
    initial_step: float = .1
    mass_smoothing: float = 1e-5
    mass_init_count: float = 4.
    target_macro_steps: float = 15.
    target_accept: float = .8
    learning_rate: float = .05
    beta1: float = .8
    beta2: float = .9
    stabilization: float = 1e-4
    learning_decay: float = .5
    init_search_limit: int = 64
    # Execution-only alternatives; mathematical and random primitives are shared.
    execution_layout: str = 'fieldwise'

    def __post_init__(self):
        if self.execution_layout not in ('fieldwise', 'packed', 'flat'):
            raise ValueError('execution_layout must be fieldwise, packed or flat')
        for name in ('chains', 'dimension', 'max_depth', 'max_halvings',
                     'min_micro_steps', 'init_search_limit'):
            x = getattr(self, name)
            if type(x) is not int or x < 1: raise ValueError(name+' must be a positive integer')
        for name in ('warmup', 'draws'):
            x = getattr(self, name)
            if type(x) is not int or x < 0: raise ValueError(name+' must be a nonnegative integer')
        if self.max_depth > 20 or self.max_halvings > 20:
            raise ValueError('depth/halving bounds must be <=20 to avoid integer overflow')
        for name in ('max_error', 'initial_step', 'mass_init_count', 'target_macro_steps',
                     'learning_rate', 'stabilization'):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0: raise ValueError(name+' must be finite and positive')
        for name in ('mass_smoothing', 'target_accept', 'beta1', 'beta2'):
            if not 0 < getattr(self, name) < 1: raise ValueError(name+' must be in (0,1)')
        if self.mass_init_count <= 1 or not 0 <= self.learning_decay <= 1:
            raise ValueError('invalid moment count or learning decay')
        largest_minimum = max(self.min_micro_steps,
            math.ceil((1 << self.max_depth)/self.target_macro_steps)+1)
        if largest_minimum*(1 << (self.max_halvings-1)) >= (1 << 30):
            raise ValueError('configured microstep counts can overflow bounded int32 loops')


class Point(NamedTuple):
    q: object
    p: object
    g: object
    lp: object
    joint: object


class Span(NamedTuple):
    bq: object
    bp: object
    bg: object
    bj: object
    fq: object
    fp: object
    fg: object
    fj: object
    q: object
    g: object
    lp: object
    weight: object


class Adam(NamedTuple):
    theta: object
    m: object
    v: object
    t: object
    b1pow: object
    b2pow: object


class Transition(NamedTuple):
    q: object
    lp: object
    g: object
    seed: object
    adam: object
    depth: object
    evaluations: object
    macros: object
    failed: object
    reverse_failed: object


class Kernel:
    def __init__(self, tf, model, config, *, tape=False):
        self.tf, self.model, self.cfg = tf, model, config
        self.c, self.d, self.tape = config.chains, config.dimension, tape
        self.dtype = tf.float64

    def f(self, x): return self.tf.constant(x, self.dtype)
    def zeros(self): return self.tf.zeros([self.c], self.dtype)
    def ints(self): return self.tf.zeros([self.c], self.tf.int32)
    def active(self): return self.tf.ones([self.c], self.tf.bool)

    def choose(self, mask, a, b):
        tf = self.tf
        if isinstance(a, tuple):
            return type(a)(*(self.choose(mask, x, y) for x, y in zip(a, b)))
        return tf.where(tf.reshape(mask, [self.c]+[1]*(a.shape.rank-1)), a, b)

    def value_gradient(self, q):
        tf = self.tf
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(q)
            lp = tf.ensure_shape(self.model(q), [self.c])
            if lp.dtype != tf.float64: raise TypeError('model must return float64 log densities')
            total = tf.reduce_sum(lp)
        g = tape.gradient(total, q)
        if g is None: raise ValueError('model must be differentiable in positions')
        return lp, tf.ensure_shape(g, [self.c, self.d])

    def health(self, q, lp, g):
        tf = self.tf
        return (tf.reduce_all(tf.math.is_finite(q), -1) & tf.math.is_finite(lp)
                & tf.reduce_all(tf.math.is_finite(g), -1))

    def random(self, seed, active, *, normal=False):
        tf = self.tf
        if self.tape:
            key, i = seed[:, 0], seed[:, 1]
            if normal:
                value = tf.cast((key[:, None]*13+tf.range(self.d)[None, :]*17)%101-50, self.dtype)/19.
            else:
                value = (tf.cast((i*73+key*19)%997, self.dtype)+.5)/997.
        else:
            fn = tf.random.stateless_normal if normal else tf.random.stateless_uniform
            value = tf.stack([fn([self.d] if normal else [], seed[i],
                dtype=self.dtype, alg='philox') for i in range(self.c)])
        seed = seed + tf.stack([self.ints(), tf.cast(active, tf.int32)], -1)
        return value, seed

    def point(self, q, p, g, lp, inv):
        return Point(q, p, g, lp, lp-.5*self.tf.reduce_sum(p*p*inv, -1))

    def span(self, p):
        return Span(p.q, p.p, p.g, p.joint, p.q, p.p, p.g, p.joint,
                    p.q, p.g, p.lp, p.joint)

    def edge(self, span, forward):
        # Only the endpoint joint density is needed to start a macro step.
        q = self.choose(forward, span.fq, span.bq)
        p = self.choose(forward, span.fp, span.bp)
        g = self.choose(forward, span.fg, span.bg)
        joint = self.choose(forward, span.fj, span.bj)
        return Point(q, p, g, self.zeros(), joint)

    def leapfrogs(self, point, inv, step, counts, active):
        tf = self.tf
        def cond(i, p): return tf.reduce_any(active & (i < counts))
        def body(i, p):
            live = active & (i < counts)
            mom = p.p + .5*step[:, None]*p.g
            q = p.q + step[:, None]*inv*mom
            # Avoid evaluating poisoned inactive rows in a batch-native model.
            q = self.choose(live, q, point.q)
            lp, g = self.value_gradient(q)
            mom += .5*step[:, None]*g
            nxt = self.point(q, mom, g, lp, inv)
            return i+1, self.choose(live, nxt, p)
        _, out = tf.while_loop(cond, body, (tf.constant(0), point), parallel_iterations=1)
        return out

    def adam_init(self, epsilon):
        tf = self.tf
        return Adam(tf.math.log(epsilon), self.zeros(), self.zeros(), self.zeros(),
                    tf.ones([self.c], self.dtype), tf.ones([self.c], self.dtype))

    def adam_observe(self, a, alpha, active):
        tf, c = self.tf, self.cfg
        t, b1, b2 = a.t+1., a.b1pow*c.beta1, a.b2pow*c.beta2
        grad = c.target_accept-alpha
        m = c.beta1*a.m+(1-c.beta1)*grad
        v = c.beta2*a.v+(1-c.beta2)*grad*grad
        theta = a.theta-(c.learning_rate/tf.pow(t, c.learning_decay))*(m/(1-b1))/(tf.sqrt(v/(1-b2))+c.stabilization)
        return self.choose(active, Adam(theta, m, v, t, b1, b2), a)

    def macro(self, start, inv, step, minimum, active, adam, adapt):
        tf, c = self.tf, self.cfg
        def cond(k, live, out, ok, revfail, ev, a):
            return (k < c.max_halvings) & tf.reduce_any(live)
        def body(k, live, out, ok, revfail, ev, a):
            factor = tf.bitwise.left_shift(tf.constant(1), k)
            count = minimum*factor
            eps = step/tf.cast(factor, self.dtype)
            trial = self.leapfrogs(start, inv, eps, count, live)
            err = tf.abs(trial.joint-start.joint)
            alpha = tf.exp(-err)
            alpha = tf.where(tf.math.is_finite(alpha), alpha, self.zeros())
            if adapt: a = self.adam_observe(a, alpha, live & (k == 0))
            accepted = live & (err <= c.max_error)
            ev += tf.where(live, count, self.ints())
            # Test all coarser reverse trajectories, each from the candidate end.
            def rcond(n, e, check, failed, cost):
                return tf.reduce_any(check & (n >= 2*minimum))
            def rbody(n, e, check, failed, cost):
                check &= n >= 2*minimum
                n, e = n//2, e*2.
                reverse = trial._replace(p=-trial.p)
                back = self.leapfrogs(reverse, inv, e, n, check)
                bad = check & (tf.abs(back.joint-trial.joint) <= c.max_error)
                return n, e, check & ~bad, failed | bad, cost+tf.where(check, n, self.ints())
            _, _, _, bad, cost = tf.while_loop(rcond, rbody,
                (count, eps, accepted, tf.zeros([self.c], tf.bool), self.ints()), parallel_iterations=1)
            out = self.choose(live, trial, out)
            return k+1, live & ~accepted, out, ok | (accepted & ~bad), revfail | bad, ev+cost, a
        _, live, out, ok, revfail, ev, adam = tf.while_loop(cond, body,
            (tf.constant(0), active, start, tf.zeros([self.c], tf.bool),
             tf.zeros([self.c], tf.bool), self.ints(), adam), parallel_iterations=1)
        return out, ok, revfail, ev, adam

    def uturn(self, old, new, forward, inv):
        bq = self.choose(forward, old.bq, new.bq)
        bp = self.choose(forward, old.bp, new.bp)
        fq = self.choose(forward, new.fq, old.fq)
        fp = self.choose(forward, new.fp, old.fp)
        delta = inv*(fq-bq)
        return (self.tf.reduce_sum(bp*delta, -1) < 0) | (self.tf.reduce_sum(fp*delta, -1) < 0)

    def combine(self, old, new, forward, seed, active, metropolis):
        tf = self.tf
        total = tf.math.reduce_logsumexp(tf.stack([old.weight, new.weight]), axis=0)
        u, seed = self.random(seed, active)
        select = tf.math.log(u) < new.weight-(old.weight if metropolis else total)
        b, f = self.choose(forward, old, new), self.choose(forward, new, old)
        chosen = self.choose(select, new, old)
        return Span(b.bq,b.bp,b.bg,b.bj,f.fq,f.fp,f.fg,f.fj,chosen.q,chosen.g,chosen.lp,total), seed

    def subtree(self, old, depth, forward, inv, eps, minimum, seed, active, adam, adapt):
        if self.cfg.execution_layout == 'flat':
            return self.subtree_flat(old, depth, forward, inv, eps, minimum, seed, active, adam, adapt)
        tf, c = self.tf, self.cfg
        stack = self.stack_init(old)
        def cond(i, live, last, stack, seed, adam, ev, macros, failed, revfail):
            return (i < tf.bitwise.left_shift(1, depth)) & tf.reduce_any(live)
        def body(i, live, last, stack, seed, adam, ev, macros, failed, revfail):
            start = self.edge(last, forward)
            p, ok, bad, cost, adam = self.macro(start, inv,
                tf.where(forward, eps, -eps), minimum, live, adam, adapt)
            failed |= live & ~ok
            revfail |= bad
            macros += tf.cast(live, tf.int32)
            live &= ok
            nxt = self.span(p)
            def cc(level, span, live, seed):
                return (level < depth) & (tf.math.floormod(i+1, tf.bitwise.left_shift(1, level+1)) == 0) & tf.reduce_any(live)
            def cb(level, span, live, seed):
                prev = self.stack_read(stack, level)
                live &= ~self.uturn(prev, span, forward, inv)
                combined, seed = self.combine(prev, span, forward, seed, live, False)
                return level+1, combined, live, seed
            level, nxt, live, seed = tf.while_loop(cc, cb, (tf.constant(0), nxt, live, seed), parallel_iterations=1)
            stack = self.stack_write(stack, level, nxt)
            return i+1, live, nxt, stack, seed, adam, ev+cost, macros, failed, revfail
        _, live, out, _, seed, adam, ev, macros, failed, revfail = tf.while_loop(cond, body,
            (tf.constant(0), active, old, stack, seed, adam, self.ints(), self.ints(),
             tf.zeros([self.c], tf.bool), tf.zeros([self.c], tf.bool)), parallel_iterations=1)
        return out, live, seed, adam, ev, macros, failed, revfail

    def pack_span(self, span):
        return self.tf.concat([x[:, None] if x.shape.rank == 1 else x for x in span], -1)

    def unpack_span(self, packed):
        widths = [self.d,self.d,self.d,1,self.d,self.d,self.d,1,self.d,self.d,1,1]
        parts = self.tf.split(packed, widths, axis=-1)
        return Span(*(x[:, 0] if i in (3,7,10,11) else x for i,x in enumerate(parts)))

    def stack_init(self, old):
        tf = self.tf
        if self.cfg.execution_layout == 'fieldwise':
            return Span(*(tf.zeros([self.cfg.max_depth+1]+x.shape.as_list(), x.dtype) for x in old))
        return tf.zeros([self.cfg.max_depth+1,self.c,8*self.d+4], self.dtype)

    def stack_read(self, stack, level):
        if self.cfg.execution_layout == 'fieldwise':
            return Span(*(x[level] for x in stack))
        return self.unpack_span(stack[level])

    def stack_write(self, stack, level, span):
        tf = self.tf
        index = tf.reshape(level, [1,1])
        if self.cfg.execution_layout == 'fieldwise':
            return Span(*(tf.tensor_scatter_nd_update(x, index, y[None]) for x,y in zip(stack,span)))
        return tf.tensor_scatter_nd_update(stack, index, self.pack_span(span)[None])

    def subtree_flat(self, old, depth, forward, inv, eps, minimum, seed, active, adam, adapt):
        """One scheduler for leaves and binary carries, preserving traversal order.

        i counts completed leaves; pending means the current span must merge
        with stack[level]. No extra random draw or numerical proposal is made.
        Macro refinement and reverse checks retain their shared implementation.
        """
        tf = self.tf
        stack = self.stack_init(old)
        def carry_needed(i, level, live):
            return ((level < depth) &
                (tf.math.floormod(i, tf.bitwise.left_shift(1, level+1)) == 0) &
                tf.reduce_any(live))
        def cond(i, level, pending, live, last, stack, seed, adam, ev, macros, failed, revfail):
            return ((i < tf.bitwise.left_shift(1, depth)) | pending) & tf.reduce_any(live)
        def body(i, level, pending, live, last, stack, seed, adam, ev, macros, failed, revfail):
            def leaf():
                p, ok, bad, cost, a = self.macro(self.edge(last, forward), inv,
                    tf.where(forward, eps, -eps), minimum, live, adam, adapt)
                return (i+1, tf.constant(0), live & ok, self.span(p), seed, a, ev+cost,
                    macros+tf.cast(live, tf.int32), failed | (live & ~ok), revfail | bad)
            def carry():
                prev = self.stack_read(stack, level)
                ok = live & ~self.uturn(prev, last, forward, inv)
                combined, rng = self.combine(prev, last, forward, seed, ok, False)
                return i, level+1, ok, combined, rng, adam, ev, macros, failed, revfail
            i, level, live, last, seed, adam, ev, macros, failed, revfail = tf.cond(pending, carry, leaf)
            pending = carry_needed(i, level, live)
            stack = tf.cond(pending, lambda:stack, lambda:self.stack_write(stack, level, last))
            return i, level, pending, live, last, stack, seed, adam, ev, macros, failed, revfail
        _, _, _, live, out, _, seed, adam, ev, macros, failed, revfail = tf.while_loop(cond, body,
            (tf.constant(0),tf.constant(0),tf.constant(False),active,old,stack,seed,adam,
             self.ints(),self.ints(),tf.zeros([self.c],tf.bool),tf.zeros([self.c],tf.bool)),
            parallel_iterations=1)
        return out, live, seed, adam, ev, macros, failed, revfail

    def transition(self, q, lp, g, eps, inv, minimum, seed, adam, *, adapt=False):
        tf, c = self.tf, self.cfg
        z, seed = self.random(seed, self.active(), normal=True)
        old = self.span(self.point(q, z/tf.sqrt(inv), g, lp, inv))
        def cond(level, live, old, seed, adam, depth, ev, macros, failed, revfail):
            return (level < c.max_depth) & tf.reduce_any(live)
        def body(level, live, old, seed, adam, depth, ev, macros, failed, revfail):
            u, seed = self.random(seed, live)
            forward = u < .5
            nxt, ok, seed, adam, cost, nm, fail, rf = self.subtree(
                old, level, forward, inv, eps, minimum, seed, live, adam, adapt)
            turn = self.uturn(old, nxt, forward, inv)
            combined, seed = self.combine(old, nxt, forward, seed, ok, True)
            return (level+1, ok & ~turn, self.choose(ok, combined, old), seed, adam,
                    depth+tf.cast(ok, tf.int32), ev+cost, macros+nm, failed|fail, revfail|rf)
        _, _, out, seed, adam, depth, ev, macros, failed, revfail = tf.while_loop(cond, body,
            (tf.constant(0), self.active(), old, seed, adam, self.ints(), self.ints(), self.ints(),
             tf.zeros([self.c], tf.bool), tf.zeros([self.c], tf.bool)), parallel_iterations=1)
        return Transition(out.q, out.lp, out.g, seed, adam, depth, ev, macros, failed, revfail)

    def initial_step(self, q, lp, g, inv, seed):
        tf, c = self.tf, self.cfg
        z, seed = self.random(seed, self.active(), normal=True)
        start = self.point(q, z/tf.sqrt(inv), g, lp, inv)
        eps = tf.fill([self.c], self.f(c.initial_step))
        counts, valid = self.ints(), self.health(q, lp, g)
        def search(eps, counts, valid, grow):
            def error(e):
                p = self.leapfrogs(start, inv, e, tf.ones([self.c], tf.int32), self.active())
                return p.joint-start.joint
            err = error(eps)
            def need(err): return err > self.f(math.log(.9)) if grow else err < self.f(math.log(.6))
            def cond(e, n, err): return tf.reduce_any(need(err) & (n < c.init_search_limit))
            def body(e, n, err):
                live = need(err) & (n < c.init_search_limit)
                e = tf.where(live, e*(2. if grow else math.sqrt(.5)), e)
                return e, n+tf.cast(live, tf.int32), error(e)
            e, n, err = tf.while_loop(cond, body, (eps, counts, err), parallel_iterations=1)
            # An infinite negative energy difference ends growth and can be
            # repaired by shrinkage; only NaN or the final nonfinite error fails.
            finite = ~tf.math.is_nan(err) if grow else tf.math.is_finite(err)
            return e, n, valid & ~need(err) & finite
        eps, counts, valid = search(eps, counts, valid, True)
        eps, counts, valid = search(eps, counts, valid, False)
        return eps, seed, valid, counts
