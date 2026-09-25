"""FAB training of the configured NeuTra IAF; TensorFlow port of author FAB.

Source: lollcat/fab-jax c9f991366ca94b2678a7ed620bc9e12655cfef1d,
sampling/{base,smc}.py, train/{fab_with_buffer,fab_without_buffer}.py,
buffer/prioritised_buffer.py. See docs/reference/neutra-fab.md for the exact
correspondence, numerical choices and deliberate departures. No map is defined
here: NeuTraTransport remains the single transport authority.

Derived from software Copyright (c) 2024 Laurence Midgley, MIT licensed.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import NamedTuple

import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, _hash

AUTHOR_REVISION = "c9f991366ca94b2678a7ed620bc9e12655cfef1d"
CHECKPOINT_SCHEMA = "bayesfilter.neutra.fab_training.v1"


def intermediate_log_prob(log_q, log_p, beta, alpha=2.):
    """Author sampling/base.py: q^(1-alpha*beta) p^(alpha*beta)."""
    a = tf.cast(alpha, log_q.dtype) * tf.cast(beta, log_q.dtype)
    return (1. - a) * log_q + a * log_p


def replay_log_correction(log_q_old, log_q, alpha=2.):
    return (1. - tf.cast(alpha, log_q.dtype)) * (
        tf.stop_gradient(log_q) - tf.stop_gradient(log_q_old))


def fab_weighted_loss(log_q, log_weights):
    """Paper Eq. 7: weighted SUM, not the author's additional factor 1/N."""
    return -tf.reduce_sum(tf.nn.softmax(tf.stop_gradient(log_weights)) * log_q)


class Point(NamedTuple):
    x: tf.Tensor
    log_q: tf.Tensor
    log_p: tf.Tensor
    grad_q: tf.Tensor
    grad_p: tf.Tensor
    valid: tf.Tensor


class Replay(NamedTuple):
    x: tf.Tensor
    log_w: tf.Tensor
    log_q_old: tf.Tensor
    index: tf.Tensor
    size: tf.Tensor


@dataclass(frozen=True)
class FABConfig:
    batch_size: int
    intermediate_distributions: int
    leapfrog_steps: int
    hmc_steps: int
    initial_step_size: float
    learning_rate: float
    beta1: float
    beta2: float
    adam_epsilon: float
    replay_capacity: int
    replay_min_size: int
    updates_per_pass: int
    correction_clip: float | None
    gradient_clip: float | None
    adapt_step_size: bool
    target_acceptance: float
    step_size_multiplier: float
    jit_compile: bool = True
    transition_operator: str = "hmc"

    def __post_init__(self):
        if self.transition_operator not in ("hmc", "metropolis"):
            raise ValueError("unknown FAB transition operator")
        for name in ("batch_size", "intermediate_distributions", "leapfrog_steps",
                     "hmc_steps", "updates_per_pass"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("FAB training must be batched")
        for name in ("initial_step_size", "learning_rate", "adam_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.:
                raise ValueError(f"invalid {name}")
        for name in ("correction_clip", "gradient_clip"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v <= 0.):
                raise ValueError(f"invalid {name}")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("invalid Adam decay")
        if not 0 < self.target_acceptance < 1 or not math.isfinite(self.step_size_multiplier) or self.step_size_multiplier <= 1:
            raise ValueError("invalid step-size adaptation")
        if type(self.replay_capacity) is not int or type(self.replay_min_size) is not int:
            raise ValueError("replay sizes must be integers")
        if self.replay_capacity:
            if not self.batch_size * self.updates_per_pass <= self.replay_min_size <= self.replay_capacity:
                raise ValueError("insufficient replay for distinct sampled minibatches")
            if self.replay_min_size % self.batch_size or self.replay_capacity % self.batch_size:
                raise ValueError("replay sizes must be whole AIS batches")
        elif self.replay_min_size or self.updates_per_pass != 1:
            raise ValueError("without replay, use exactly one update per fresh AIS pass")


class FABTrainer:
    """AIS, author HMC or Gaussian Metropolis, optional corrected replay.

    target_value_score accepts a native [batch, dimension] FP64 tensor and
    returns values, first scores and boolean validity. All repeated numerical
    work is in stable-signature graphs. Python orchestrates outer iterations
    and artifacts; no Python sample or leapfrog loop is used.
    """

    def __init__(self, transport, target_value_score, config, *, target_signature, seed):
        if not isinstance(transport, NeuTraTransport) or not transport.trainable_variables:
            raise ValueError("a trainable configured transport is required")
        c = transport.config
        if (c.kind, c.mask_policy, c.activation, c.scale_transform, c.iaf_initializer,
                c.permutation_policy) != ("iaf", "hoffman_block_masks_v1", "elu",
                "neutra_conditional_tanh", "hoffman_variance_scaling", "full_reverse"):
            raise ValueError("FAB requires the canonical author IAF profile")
        if c.stages != 3 or c.iaf_variance_scale != .02:
            raise ValueError("canonical IAF stage/initializer profile changed")
        if not isinstance(target_signature, str) or len(target_signature) != 64 or any(x not in "0123456789abcdef" for x in target_signature):
            raise ValueError("target_signature must be SHA256 hex")
        if len(seed) != 2 or any(type(s) is not int or not 0 <= s < 2**31 for s in seed):
            raise ValueError("seed must contain two nonnegative int32 values")
        self.transport, self.target, self.config = transport, target_value_score, config
        self.target_signature, self.seed = target_signature, tuple(seed)
        self.dtype, self.dimension = transport.dtype, transport.parameter_dim
        self.variables = transport.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=config.beta1,
            beta_2=config.beta2, epsilon=config.adam_epsilon)
        self.optimizer.build(self.variables)
        self.steps = tf.fill([config.intermediate_distributions], tf.constant(config.initial_step_size, self.dtype))
        self.betas = tf.linspace(tf.cast(0., self.dtype), tf.cast(1., self.dtype), config.intermediate_distributions + 2)
        self.pass_index = 0
        self.replay = None
        b, d = config.batch_size, self.dimension
        spec = [tf.TensorSpec([b, d], self.dtype), tf.TensorSpec([2], tf.int32),
                tf.TensorSpec([config.intermediate_distributions], self.dtype)]
        self.ais = tf.function(self._ais, input_signature=spec, jit_compile=config.jit_compile, autograph=False)
        self.draw = tf.function(self._draw, input_signature=[tf.TensorSpec([2], tf.int32)],
                                jit_compile=config.jit_compile, autograph=False)
        self.update = tf.function(self._update, input_signature=[tf.TensorSpec([b, d], self.dtype),
            tf.TensorSpec([b], self.dtype), tf.TensorSpec([b], self.dtype), tf.TensorSpec([], tf.bool)],
            jit_compile=config.jit_compile, autograph=False)
        if config.replay_capacity:
            capacity = config.replay_capacity
            rs = Replay(tf.TensorSpec([capacity, d], self.dtype), tf.TensorSpec([capacity], self.dtype),
                tf.TensorSpec([capacity], self.dtype), tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32))
            self.replay_sample = tf.function(self._replay_sample,
                input_signature=[rs, tf.TensorSpec([2], tf.int32)], jit_compile=config.jit_compile, autograph=False)
            self.replay_add = tf.function(self._replay_add, input_signature=[rs,
                tf.TensorSpec([b, d], self.dtype), tf.TensorSpec([b], self.dtype), tf.TensorSpec([b], self.dtype)],
                jit_compile=config.jit_compile, autograph=False)
            self.replay_adjust = tf.function(self._replay_adjust, input_signature=[rs,
                tf.TensorSpec([b], tf.int32), tf.TensorSpec([b], self.dtype), tf.TensorSpec([b], self.dtype)],
                jit_compile=config.jit_compile, autograph=False)

    def _key(self, role):
        return tf.random.experimental.stateless_fold_in(tf.constant(self.seed, tf.int32),
            tf.constant(4 * self.pass_index + role, tf.int32))

    def _draw(self, key):
        z = tf.random.stateless_normal([self.config.batch_size, self.dimension], key, dtype=self.dtype)
        return self.transport.forward_and_logdet(z)[0]

    def _point(self, x):
        if self.config.transition_operator == "metropolis":
            log_q = self.transport.log_prob(x)
            grad_q = tf.zeros_like(x)  # Unused by the gradient-free mutation.
        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x)
                log_q = self.transport.log_prob(x)
            grad_q = tape.gradient(log_q, x)
        log_p, grad_p, valid = self.target(tf.cast(x, tf.float64))
        log_p, grad_p = tf.cast(log_p, self.dtype), tf.cast(grad_p, self.dtype)
        finite = (tf.reduce_all(tf.math.is_finite(x), -1) & tf.math.is_finite(log_q)
            & tf.math.is_finite(log_p) & tf.reduce_all(tf.math.is_finite(grad_q), -1)
            & tf.reduce_all(tf.math.is_finite(grad_p), -1))
        return Point(x, log_q, log_p, grad_q, grad_p, tf.cast(valid, tf.bool) & finite)

    def _log_prob(self, point, beta):
        return intermediate_log_prob(point.log_q, point.log_p, beta)

    def _score(self, point, beta):
        return intermediate_log_prob(point.grad_q, point.grad_p, beta)

    def _hmc(self, point, key, beta, step_size):
        keys = tf.random.experimental.stateless_split(key, 2)
        momentum = tf.random.stateless_normal(tf.shape(point.x), keys[0], dtype=self.dtype)
        initial_momentum = momentum
        momentum += .5 * step_size * self._score(point, beta)

        def body(i, proposal, p, path_valid):
            proposal = self._point(proposal.x + step_size * p)
            p += tf.where(i + 1 == self.config.leapfrog_steps, tf.cast(.5, self.dtype), tf.cast(1., self.dtype)) * step_size * self._score(proposal, beta)
            return i + 1, proposal, p, path_valid & proposal.valid

        _, proposal, momentum, path_valid = tf.while_loop(lambda i, *_: i < self.config.leapfrog_steps,
            body, (tf.constant(0), point, momentum, point.valid), parallel_iterations=1)
        log_accept = (self._log_prob(proposal, beta) - self._log_prob(point, beta)
            + .5 * tf.reduce_sum(initial_momentum**2 - momentum**2, -1))
        valid = path_valid & tf.math.is_finite(log_accept)
        log_accept = tf.where(valid, tf.minimum(log_accept, 0.), tf.cast(-math.inf, self.dtype))
        uniform = tf.random.stateless_uniform([self.config.batch_size], keys[1], dtype=self.dtype)
        accepted = tf.math.log(uniform) < log_accept
        kept = Point(*(tf.where(accepted[:, None] if new.shape.rank == 2 else accepted, new, old)
                       for new, old in zip(proposal, point)))
        return kept, tf.reduce_mean(tf.exp(log_accept)), tf.reduce_mean(tf.cast(accepted, self.dtype)), tf.reduce_sum(tf.cast(~valid, tf.int32))

    def _metropolis(self, point, key, beta, step_size):
        """Author metropolis.py: symmetric Gaussian random walk and MH ratio."""
        keys = tf.random.experimental.stateless_split(key, 2)
        noise = tf.random.stateless_normal(tf.shape(point.x), keys[0], dtype=self.dtype)
        proposal = self._point(point.x + step_size * noise)
        log_accept = self._log_prob(proposal, beta) - self._log_prob(point, beta)
        valid = point.valid & proposal.valid & tf.math.is_finite(log_accept)
        log_accept = tf.where(valid, tf.minimum(log_accept, 0.), tf.cast(-math.inf, self.dtype))
        uniform = tf.random.stateless_uniform([self.config.batch_size], keys[1], dtype=self.dtype)
        accepted = tf.math.log(uniform) < log_accept
        kept = Point(*(tf.where(accepted[:, None] if new.shape.rank == 2 else accepted, new, old)
                       for new, old in zip(proposal, point)))
        return kept, tf.reduce_mean(tf.exp(log_accept)), tf.reduce_mean(tf.cast(accepted, self.dtype)), tf.reduce_sum(tf.cast(~valid, tf.int32))

    def _ais(self, x, key, steps):
        point = self._point(x)
        initial_valid = tf.reduce_all(point.valid)
        # Author ordering: weight beta0->beta1, then mutate at betak and
        # increment to beta(k+1). There is no final beta=1 mutation.
        log_w = self._log_prob(point, self.betas[1]) - self._log_prob(point, self.betas[0])
        acc = tf.zeros_like(steps)
        movement = tf.zeros_like(steps)
        invalid = tf.zeros_like(steps, dtype=tf.int32)

        def stage(k, point, log_w, steps, acc, movement, invalid):
            stage_key = tf.random.experimental.stateless_fold_in(key, k)
            beta = self.betas[k + 1]
            def transition(j, point, epsilon, a_sum, moved_sum, bad_sum):
                kernel = self._hmc if self.config.transition_operator == "hmc" else self._metropolis
                point, a, moved, bad = kernel(point,
                    tf.random.experimental.stateless_fold_in(stage_key, j), beta, epsilon)
                if self.config.adapt_step_size and self.config.transition_operator == "hmc":
                    epsilon *= tf.where(a > self.config.target_acceptance,
                        tf.constant(self.config.step_size_multiplier, self.dtype),
                        tf.constant(1. / self.config.step_size_multiplier, self.dtype))
                return j + 1, point, epsilon, a_sum + a, moved_sum + moved, bad_sum + bad
            _, point, epsilon, a, moved, bad = tf.while_loop(
                lambda j, *_: j < self.config.hmc_steps, transition,
                (tf.constant(0), point, steps[k], tf.cast(0., self.dtype), tf.cast(0., self.dtype), tf.constant(0)),
                parallel_iterations=1)
            if self.config.adapt_step_size and self.config.transition_operator == "metropolis":
                # Author Metropolis holds epsilon fixed for all mutations at
                # this temperature, then adapts once using mean acceptance.
                epsilon *= tf.where(a / self.config.hmc_steps > self.config.target_acceptance,
                    tf.constant(self.config.step_size_multiplier, self.dtype),
                    tf.constant(1. / self.config.step_size_multiplier, self.dtype))
            log_w += self._log_prob(point, self.betas[k + 2]) - self._log_prob(point, beta)
            index = tf.reshape(k, [1, 1])
            return (k + 1, point, log_w, tf.tensor_scatter_nd_update(steps, index, [epsilon]),
                tf.tensor_scatter_nd_update(acc, index, [a / self.config.hmc_steps]),
                tf.tensor_scatter_nd_update(movement, index, [moved / self.config.hmc_steps]),
                tf.tensor_scatter_nd_update(invalid, index, [bad]))
        _, point, log_w, steps, acc, movement, invalid = tf.while_loop(
            lambda k, *_: k < self.config.intermediate_distributions, stage,
            (tf.constant(0), point, log_w, steps, acc, movement, invalid), parallel_iterations=1)
        weights = tf.nn.softmax(log_w)
        valid = initial_valid & tf.reduce_all(point.valid) & tf.reduce_all(tf.math.is_finite(log_w))
        return {"x": point.x, "log_w": log_w, "log_q": point.log_q, "valid": valid,
            "steps": steps, "acceptance": acc, "movement": movement, "invalid_proposals": invalid,
            "ess_fraction": 1. / (self.config.batch_size * tf.reduce_sum(weights**2)),
            "max_weight": tf.reduce_max(weights),
            "log_mean_weight": tf.reduce_logsumexp(log_w) - tf.math.log(tf.cast(self.config.batch_size, self.dtype))}

    def _update(self, x, log_w, old_q, from_replay):
        x = tf.stop_gradient(x)
        with tf.GradientTape() as tape:
            log_q = self.transport.log_prob(x)
            adjustment = replay_log_correction(old_q, log_q)
            maximum = tf.constant(math.inf if self.config.correction_clip is None else math.log(self.config.correction_clip), self.dtype)
            correction = tf.exp(tf.minimum(adjustment, maximum))
            loss = tf.cond(from_replay, lambda: -tf.reduce_mean(correction * log_q),
                lambda: fab_weighted_loss(log_q, log_w))
        grads = tape.gradient(loss, self.variables)
        norm = tf.linalg.global_norm(grads)
        clipped = grads if self.config.gradient_clip is None else tf.clip_by_global_norm(grads, self.config.gradient_clip, use_norm=norm)[0]
        finite = (tf.math.is_finite(loss) & tf.math.is_finite(norm) & tf.reduce_all(tf.math.is_finite(log_q))
                  & tf.reduce_all(tf.math.is_finite(adjustment)))
        saved = tuple(tf.identity(v) for v in (*self.variables, *self.optimizer.variables))
        def apply():
            self.optimizer.apply_gradients(zip(clipped, self.variables))
            new_q = self.transport.log_prob(x)
            all_vars = (*self.variables, *self.optimizer.variables)
            valid = tf.reduce_all(tf.math.is_finite(new_q)) & tf.reduce_all(tf.stack([
                tf.reduce_all(tf.math.is_finite(v)) for v in all_vars if tf.as_dtype(v.dtype).is_floating]))
            def rollback():
                for v, old in zip(all_vars, saved):
                    v.assign(old)
                return tf.constant(False)
            return tf.cond(valid, lambda: tf.constant(True), rollback)
        valid = tf.cond(finite, apply, lambda: tf.constant(False))
        return {"valid": valid, "loss": loss, "gradient_norm": norm,
            "gradient_clipped": norm > tf.cast(math.inf if self.config.gradient_clip is None else self.config.gradient_clip, self.dtype),
            "correction_clipped_fraction": tf.reduce_mean(tf.cast(adjustment > maximum, self.dtype)),
            "log_w_adjustment": adjustment, "log_q": log_q,
            "iteration": tf.identity(self.optimizer.iterations)}

    def _replay_sample(self, replay, key):
        # Author Gumbel top-k without replacement, then random permutation.
        keys = tf.random.experimental.stateless_split(key, 2)
        uniform = tf.random.stateless_uniform(tf.shape(replay.log_w), keys[0], dtype=self.dtype)
        gumbel = -tf.math.log(-tf.math.log(uniform))
        indices = tf.math.top_k(replay.log_w + gumbel,
            k=self.config.batch_size * self.config.updates_per_pass).indices
        return tf.random.experimental.stateless_shuffle(indices, keys[1])

    def _replay_add(self, replay, x, log_w, log_q):
        indices = tf.reshape((tf.range(self.config.batch_size) + replay.index) % self.config.replay_capacity, [-1, 1])
        return Replay(tf.tensor_scatter_nd_update(replay.x, indices, x),
            tf.tensor_scatter_nd_update(replay.log_w, indices, log_w),
            tf.tensor_scatter_nd_update(replay.log_q_old, indices, log_q),
            (replay.index + self.config.batch_size) % self.config.replay_capacity,
            tf.minimum(replay.size + self.config.batch_size, self.config.replay_capacity))

    def _replay_adjust(self, replay, indices, log_q, adjustment):
        indices2 = tf.reshape(indices, [-1, 1])
        return Replay(replay.x, tf.tensor_scatter_nd_update(replay.log_w, indices2,
            tf.gather(replay.log_w, indices) + adjustment),
            tf.tensor_scatter_nd_update(replay.log_q_old, indices2, log_q), replay.index, replay.size)

    def step(self, *, train=True):
        """One fresh AIS pass and its declared updates; host boundary validates."""
        result = self.ais(self.draw(self._key(0)), self._key(1), self.steps)
        if not bool(result["valid"].numpy()):
            raise ValueError("invalid FAB AIS pass; initial q draws must not be silently conditioned")
        self.steps = result["steps"]
        updates = []
        if self.config.replay_capacity:
            if self.replay is None:
                capacity = self.config.replay_capacity
                self.replay = Replay(tf.zeros([capacity, self.dimension], self.dtype),
                    tf.fill([capacity], tf.cast(-math.inf, self.dtype)), tf.zeros([capacity], self.dtype),
                    tf.constant(0), tf.constant(0))
            # The new pass uses pre-update q; do not insert it until after the
            # replay updates, matching the author JAX iteration ordering.
            # fab_with_buffer.init fills floor(minimum/batch) + 1 batches.
            # Preserve that extra batch even when the minimum divides exactly.
            initial_passes = self.config.replay_min_size // self.config.batch_size + 1
            if (train and self.pass_index >= initial_passes
                    and int(self.replay.size.numpy()) >= self.config.replay_min_size):
                indices = self.replay_sample(self.replay, self._key(2))
                for j in range(self.config.updates_per_pass):
                    chosen = indices[j*self.config.batch_size:(j+1)*self.config.batch_size]
                    update = self.update(tf.gather(self.replay.x, chosen), tf.gather(self.replay.log_w, chosen),
                        tf.gather(self.replay.log_q_old, chosen), tf.constant(True))
                    if not bool(update["valid"].numpy()):
                        raise ValueError("invalid FAB replay update (optimizer rolled back)")
                    self.replay = self.replay_adjust(self.replay, chosen, update["log_q"], update["log_w_adjustment"])
                    updates.append(update)
            self.replay = self.replay_add(self.replay, result["x"], result["log_w"], result["log_q"])
        elif train:
            update = self.update(result["x"], result["log_w"], result["log_q"], tf.constant(False))
            if not bool(update["valid"].numpy()):
                raise ValueError("invalid FAB update (optimizer rolled back)")
            updates.append(update)
        self.pass_index += 1
        return {**result, "updates": updates, "pass_index": self.pass_index}

    def checkpoint(self):
        replay = None if self.replay is None else {k: v.numpy().tolist() for k, v in self.replay._asdict().items()}
        # Unoccupied log weights are represented by null at the JSON boundary.
        if replay is not None:
            replay["log_w"] = [v if math.isfinite(v) else None for v in replay["log_w"]]
        body = {"schema": CHECKPOINT_SCHEMA, "author_revision": AUTHOR_REVISION,
            "target_signature": self.target_signature, "transport_config": self.transport.config.payload(),
            "config": asdict(self.config), "seed": list(self.seed), "pass_index": self.pass_index,
            "steps": self.steps.numpy().tolist(), "parameters": self.transport.parameter_state(),
            "optimizer": [{"shape": list(v.shape), "dtype": tf.as_dtype(v.dtype).name,
                           "value": v.numpy().tolist()} for v in self.optimizer.variables], "replay": replay}
        return {**body, "checkpoint_hash": _hash(body)}

    def restore(self, checkpoint):
        body = {k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}
        if checkpoint.get("checkpoint_hash") != _hash(body):
            raise ValueError("FAB checkpoint hash mismatch")
        expected = (CHECKPOINT_SCHEMA, AUTHOR_REVISION, self.target_signature, self.transport.config.payload(), asdict(self.config), list(self.seed))
        actual = tuple(checkpoint[k] for k in ("schema", "author_revision", "target_signature", "transport_config", "config", "seed"))
        if expected != actual or len(checkpoint["optimizer"]) != len(self.optimizer.variables):
            raise ValueError("FAB checkpoint configuration/target mismatch")
        pending = []
        for var, row in zip(self.optimizer.variables, checkpoint["optimizer"]):
            value = tf.convert_to_tensor(row["value"], var.dtype)
            if row["shape"] != list(var.shape) or row["dtype"] != tf.as_dtype(var.dtype).name or value.shape != var.shape:
                raise ValueError("invalid FAB optimizer checkpoint shape/dtype")
            if tf.as_dtype(var.dtype).is_floating and not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
                raise ValueError("nonfinite FAB optimizer checkpoint")
            pending.append((var, value))
        steps = tf.convert_to_tensor(checkpoint["steps"], self.dtype)
        if steps.shape != self.steps.shape or not bool(tf.reduce_all(tf.math.is_finite(steps) & (steps > 0.))):
            raise ValueError("invalid FAB step-size checkpoint")
        index = checkpoint["pass_index"]
        if type(index) is not int or index < 0:
            raise ValueError("invalid FAB pass counter")
        replay = None
        if checkpoint["replay"] is not None:
            r = checkpoint["replay"]
            replay = Replay(tf.convert_to_tensor(r["x"], self.dtype),
                tf.convert_to_tensor([(-math.inf if w is None else w) for w in r["log_w"]], self.dtype),
                tf.convert_to_tensor(r["log_q_old"], self.dtype), tf.constant(r["index"], tf.int32), tf.constant(r["size"], tf.int32))
            n = self.config.replay_capacity
            if replay.x.shape != (n, self.dimension) or replay.log_w.shape != (n,) or replay.log_q_old.shape != (n,):
                raise ValueError("invalid FAB replay shape")
            if not 0 <= r["index"] < n or not 0 <= r["size"] <= n:
                raise ValueError("invalid FAB replay counters")
            if not bool(tf.reduce_all(tf.math.is_finite(replay.x))) or not bool(tf.reduce_all(tf.math.is_finite(replay.log_q_old))):
                raise ValueError("invalid FAB replay values")
            if int(tf.reduce_sum(tf.cast(tf.math.is_finite(replay.log_w), tf.int32))) != r["size"]:
                raise ValueError("invalid FAB replay weight count")
        self.transport.restore_parameters(checkpoint["parameters"])
        for var, value in pending:
            var.assign(value)
        self.steps, self.pass_index, self.replay = steps, index, replay
