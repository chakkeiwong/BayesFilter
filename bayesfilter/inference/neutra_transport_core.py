"""Single numerical authority for NeuTra transports and path gradients.

Public configuration lives in neutra_transport; historical classes translate
their saved conventions to these kernels. Static Python loops traverse network
layers only. Batch rows stay in native tensor operations; coordinate derivative
and inverse loops have a single TensorFlow body, with no pfor.
Source/derivation: docs/reference/neutra-implementation.md.
"""
from __future__ import annotations

import math
import tensorflow as tf

AUTHORITY = "bayesfilter.neutra.transport_core.v1"


def activation(values, name):
    if name == "elu":
        return tf.nn.elu(values)
    if name == "tanh":
        return tf.math.tanh(values)
    if name == "relu":
        return tf.nn.relu(values)
    raise ValueError(f"unsupported activation: {name}")


def activation_derivative(values, name):
    if name == "elu":
        return tf.where(values > 0., tf.ones_like(values), tf.exp(values))
    if name == "tanh":
        return 1. - tf.square(tf.math.tanh(values))
    if name == "relu":
        return tf.cast(values > 0., values.dtype)
    raise ValueError(f"unsupported activation: {name}")


def dense_masks(dimension, hidden_layers, *, outputs_per_dimension=2, dtype=tf.float64):
    """Legacy MADE degrees, grouped output layout; strict dependency on x_<j."""
    degrees = [list(range(1, dimension + 1))]
    for width in hidden_layers:
        degrees.append([1 + i % max(1, dimension - 1) for i in range(width)])
    degrees.append(list(range(1, dimension + 1)) * outputs_per_dimension)
    return tuple(tf.constant([
        [float(a < b if i == len(degrees)-2 else a <= b) for b in right]
        for a in left], dtype)
        for i, (left, right) in enumerate(zip(degrees[:-1], degrees[1:])))


def strict_autoregressive_mask(dimension, *, dtype=tf.float64):
    return tf.constant([[float(i < j) for j in range(dimension)]
                        for i in range(dimension)], dtype)


def hoffman_masks(dimension, hidden_layers, *, dtype=tf.float64):
    """TFP masked_dense blocks: exclusive first, inclusive remaining layers.

    NeuTra author code uses separate scale/shift output heads when pre-clipping.
    Grouped concatenation here is exactly those two heads in one matrix.
    """
    if any(width % dimension for width in hidden_layers):
        raise ValueError("author block-mask widths must be multiples of dimension")
    degrees = [list(range(dimension))]
    degrees.extend([[i//(width//dimension) for i in range(width)] for width in hidden_layers])
    degrees.append(list(range(dimension))*2)
    return tuple(tf.constant([[float(a < b if index == 0 else a <= b) for b in right]
        for a in left], dtype) for index, (left, right) in enumerate(zip(degrees[:-1], degrees[1:])))


def masked_network(stage, values):
    hidden, cache = values, []
    for weight, bias, mask in zip(stage.weights[:-1], stage.biases[:-1], stage.masks[:-1]):
        pre = tf.matmul(hidden, weight * mask) + bias
        cache.append(pre)
        hidden = activation(pre, stage.activation)
    return (tf.matmul(hidden, stage.weights[-1] * stage.masks[-1]) + stage.biases[-1],
            tuple(cache))


def masked_pullback(stage, cotangent, cache):
    cotangent = tf.matmul(cotangent, stage.weights[-1]*stage.masks[-1], transpose_b=True)
    for i in reversed(range(len(cache))):
        cotangent *= activation_derivative(cache[i], stage.activation)
        cotangent = tf.matmul(cotangent, stage.weights[i]*stage.masks[i], transpose_b=True)
    return cotangent


def _dimension(stage):
    return stage.dimension if hasattr(stage, "dimension") else stage.dim


def _linear(stage, name, values):
    weight = getattr(stage, name, None)
    if weight is None:
        return tf.zeros_like(values)
    return tf.matmul(values, weight * stage.scale_linear_skip_mask)


def _linear_pullback(stage, name, cotangent):
    weight = getattr(stage, name, None)
    if weight is None:
        return tf.zeros_like(cotangent)
    return tf.matmul(cotangent, weight*stage.scale_linear_skip_mask, transpose_b=True)


def _anchor_weight(stage):
    weight = getattr(stage, "anchor_shift_weight", None)
    return None if weight is None else (tf.linalg.band_part(weight, -1, 0)
                                      - tf.linalg.diag(tf.linalg.diag_part(weight)))


def iaf_parameters(stage, values):
    """scale, shift, dscale/dlogit, hidden cache, logits; preserves old maps."""
    raw, cache = masked_network(stage, values)
    d = _dimension(stage)
    logits = raw[..., :d] + _linear(stage, "scale_linear_skip_weight", values)
    shift = raw[..., d:]
    anchor = _anchor_weight(stage)
    if anchor is not None:
        logits += stage.anchor_scale_raw
        shift += tf.matmul(values, anchor, transpose_b=True) + stage.anchor_shift_bias
    convention = getattr(stage, "scale_transform", "bounded_tanh")
    if convention == "identity":
        scale, derivative = logits, tf.ones_like(logits)
    elif convention == "dsge_bounded_tanh":
        tangent = tf.math.tanh(logits)
        scale, derivative = stage.s_max*tangent, stage.s_max*(1.-tf.square(tangent))
    elif convention in ("bounded_tanh", "neutra_conditional_tanh"):
        # Author-code option: no bias in h; a free bias AFTER the conditional cap.
        bias = stage.biases[-1][..., :d] if convention == "neutra_conditional_tanh" else 0.
        tangent = tf.math.tanh((logits-bias)/stage.s_max)
        scale, derivative = bias + stage.s_max*tangent, 1.-tf.square(tangent)
    else:
        raise ValueError(f"unknown IAF scale convention: {convention}")
    scale += _linear(stage, "unbounded_scale_linear_weight", values)
    return scale, shift, derivative, cache, logits


def iaf_network(stage, values):
    return iaf_parameters(stage, values)[:2]


def iaf_network_cache(stage, values):
    return iaf_parameters(stage, values)[:4]


def iaf_network_diagnostics(stage, values):
    scale, shift, _, cache, logits = iaf_parameters(stage, values)
    width = max(stage.hidden_layers, default=0)
    hidden = (tf.stack([tf.pad(v, [[0, 0], [0, width-int(v.shape[-1])]]) for v in cache], axis=1)
              if cache else tf.zeros((tf.shape(values)[0], 0, width), values.dtype))
    return scale, shift, logits, hidden


def iaf_forward(stage, values):
    scale, shift = iaf_network(stage, values)
    return iaf_apply(values, scale, shift)


def iaf_apply(values, scale, shift):
    return values*tf.exp(scale)+shift, tf.reduce_sum(scale, axis=-1)


def iaf_inverse(stage, output):
    dimension = _dimension(stage)
    def solve(index, values):
        scale, shift = iaf_network(stage, values)
        x = (output[..., index]-shift[..., index])*tf.exp(-scale[..., index])
        return index+1, values+(x-values[..., index])[..., None]*tf.one_hot(index, dimension, dtype=values.dtype)
    return tf.while_loop(lambda i, _: i < dimension, solve,
        (tf.constant(0), tf.zeros_like(output)), parallel_iterations=1,
        maximum_iterations=dimension)[1]


def iaf_inverse_logdet(stage, output):
    values = iaf_inverse(stage, output)
    return values, tf.reduce_sum(iaf_network(stage, values)[0], axis=-1)


def iaf_network_pullback(stage, cotangent, cache):
    return masked_pullback(stage, cotangent, cache) + _linear_pullback(
        stage, "scale_linear_skip_weight", cotangent[..., :_dimension(stage)])


def iaf_pullback(stage, values, score):
    scale, _, derivative, cache, _ = iaf_parameters(stage, values)
    scale_cotangent = score*values*tf.exp(scale)
    raw_cotangent = tf.concat((scale_cotangent*derivative, score), axis=-1)
    result = score*tf.exp(scale)+iaf_network_pullback(stage, raw_cotangent, cache)
    result += _linear_pullback(stage, "unbounded_scale_linear_weight", scale_cotangent)
    anchor = _anchor_weight(stage)
    return result if anchor is None else result+tf.matmul(score, anchor)


def iaf_logdet_score(stage, values):
    _, _, derivative, cache, _ = iaf_parameters(stage, values)
    raw = tf.concat((derivative, tf.zeros_like(derivative)), axis=-1)
    return iaf_network_pullback(stage, raw, cache) + _linear_pullback(
        stage, "unbounded_scale_linear_weight", tf.ones_like(derivative))


def affine_forward(values, shift, scale, *, logdet=None, matrix=None, transpose=True):
    output = (values*scale if matrix is None else tf.matmul(values, matrix, transpose_b=transpose))+shift
    if logdet is None:
        logdet = tf.reduce_sum(tf.math.log(tf.abs(scale))) if matrix is None else tf.linalg.slogdet(matrix)[1]
    return output, tf.zeros(tf.shape(values)[:-1], values.dtype)+logdet


def affine_inverse(values, shift, scale, *, logdet=None, matrix=None, transpose=True):
    centered = values-shift
    if matrix is None:
        output = centered/scale
    else:
        output = tf.transpose(tf.linalg.solve(matrix if transpose else tf.transpose(matrix), tf.transpose(centered)))
    if logdet is None:
        logdet = tf.reduce_sum(tf.math.log(tf.abs(scale))) if matrix is None else tf.linalg.slogdet(matrix)[1]
    return output, tf.zeros(tf.shape(values)[:-1], values.dtype)+logdet


def affine_pullback(score, scale=None, *, matrix=None, transpose=True):
    return score*scale if matrix is None else tf.matmul(score, matrix, transpose_b=not transpose)


def interleaved_stages(stages, dimension, policy="full_reverse"):
    permutation = Permutation(dimension, policy)
    return tuple(item for i, stage in enumerate(stages)
                 for item in ((stage, permutation) if i+1 < len(stages) else (stage,)))


def compose_inverse(components, values):
    for component in reversed(components):
        if hasattr(component, "inverse"):
            values = component.inverse(values)
        else:
            values = component.inverse_and_forward_logdet(values)[0]
    return values


def compose_forward(components, values):
    logdet = tf.zeros(tf.shape(values)[:-1], values.dtype)
    for component in components:
        values, increment = component.forward_and_logdet(values)
        logdet += increment
    return values, logdet


def compose_pullback(components, values, score=None):
    inputs = []
    for component in components:
        inputs.append(values)
        values, _ = component.forward_and_logdet(values)
    include_logdet = score is None
    if include_logdet:
        score = tf.zeros_like(values)
    for component, values in reversed(tuple(zip(components, inputs))):
        score = component.pullback_score(values, score)
        if include_logdet:
            score += component.logdet_score(values)
    return score


def sigmoid_mixture(x, log_slopes, offsets, weight_logits):
    """Huang Eq.(8), full real support, stable value/logderivative/score.

    Leading axes broadcast. The final parameter axis indexes sigmoid units.
    The scalar score is d log(dy/dx)/dx with pseudo-parameters held fixed.
    """
    slopes = tf.exp(log_slopes)
    log_w = tf.nn.log_softmax(weight_logits, axis=-1)
    u = x[..., None]*slopes+offsets
    log_p, log_c = -tf.nn.softplus(-u), -tf.nn.softplus(u)
    log_s = tf.reduce_logsumexp(log_w+log_p, axis=-1)
    log_complement = tf.reduce_logsumexp(log_w+log_c, axis=-1)
    terms = log_w+log_slopes+log_p+log_c
    log_n = tf.reduce_logsumexp(terms, axis=-1)
    score = (tf.reduce_sum(tf.nn.softmax(terms, axis=-1)*slopes*(1.-2.*tf.math.sigmoid(u)), axis=-1)
             -tf.exp(log_n-log_s)+tf.exp(log_n-log_complement))
    valid = tf.reduce_all(tf.math.is_finite(slopes) & (slopes > 0.)
                         & tf.math.is_finite(offsets) & tf.math.is_finite(log_w)
                         & (tf.exp(log_w) > 0.), axis=-1)
    nan = tf.constant(float("nan"), x.dtype)
    return tuple(tf.where(valid, v, nan) for v in
                 (log_s-log_complement, log_n-log_s-log_complement, score))


def sigmoid_inverse(y, log_slopes, offsets, logits, *, atol, rtol, max_iterations):
    # Each sigmoid equals sigmoid(y) at its crossing; their extrema bracket
    # the root of the positive weighted mixture. No guessed search radius.
    crossings = (y[..., None]-offsets)*tf.exp(-log_slopes)
    lo, hi = tf.reduce_min(crossings, axis=-1), tf.reduce_max(crossings, axis=-1)
    tolerance = tf.constant(atol, y.dtype)+tf.constant(rtol, y.dtype)*tf.abs(y)
    def evaluate(x, lower, upper):
        value, ld, _ = sigmoid_mixture(x, log_slopes, offsets, logits)
        residual = value-y
        xtol = tf.constant(atol, y.dtype)+tf.constant(rtol, y.dtype)*tf.abs(x)
        valid = (tf.math.is_finite(x) & tf.math.is_finite(ld) & tf.math.is_finite(residual)
                 & (tf.abs(residual) <= tolerance) & ((upper-lower) <= 2.*xtol))
        return residual, ld, valid
    def condition(i, lower, upper, x):
        return (i < max_iterations) & ~tf.reduce_all(evaluate(x, lower, upper)[2])
    def body(i, lower, upper, x):
        residual, _, done = evaluate(x, lower, upper)
        lower = tf.where((residual < 0.) & ~done, x, lower)
        upper = tf.where((residual >= 0.) & ~done, x, upper)
        return i+1, lower, upper, tf.where(done, x, .5*lower+.5*upper)
    iterations, lo, hi, solution = tf.while_loop(condition, body,
        (tf.constant(0), lo, hi, .5*lo+.5*hi), parallel_iterations=1,
        maximum_iterations=max_iterations)
    _, ld, valid = evaluate(solution, lo, hi)
    return solution, ld, valid, iterations


def value_jacobian_logdet_score(layer, values):
    """One forward evaluation; coordinate VJPs with one traced body, no pfor."""
    dimension = int(values.shape[-1])
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(values)
        output, ld = layer.forward_and_logdet(values)
    ld_score = tape.gradient(ld, values, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    rows = tf.TensorArray(values.dtype, size=dimension, element_shape=values.shape)
    def body(i, result):
        cotangent = tf.broadcast_to(tf.one_hot(i, dimension, dtype=values.dtype), tf.shape(output))
        row = tape.gradient(output, values, output_gradients=cotangent,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return i+1, result.write(i, row)
    _, rows = tf.while_loop(lambda i, _: i < dimension, body,
        (tf.constant(0), rows), parallel_iterations=1, maximum_iterations=dimension)
    return output, ld, tf.transpose(rows.stack(), (1, 0, 2)), ld_score


def _score_layers(transport):
    if hasattr(transport, "score_layers"):
        return transport.score_layers
    if hasattr(transport, "first") and hasattr(transport, "second"):
        return (*_score_layers(transport.first), *_score_layers(transport.second))
    if hasattr(transport, "components"):
        return tuple(child for c in transport.components for child in _score_layers(c))
    if hasattr(transport, "children"):
        return tuple(child for c in transport.children for child in _score_layers(c))
    if hasattr(transport, "inner") and hasattr(transport, "center") and hasattr(transport, "scale"):
        return (*_score_layers(transport.inner), AffineLayer(transport.center, transport.scale))
    if hasattr(transport, "layers") and hasattr(transport, "affine_factor"):
        return (*interleaved_stages(transport.layers, transport.dimension),
                AffineLayer(transport.affine_center, matrix=transport.affine_factor))
    if hasattr(transport, "local_transport") and hasattr(transport, "_lchol"):
        return (*_score_layers(transport.local_transport), AffineLayer(transport._mu, matrix=transport._lchol))
    return (transport,)


def forward_proposal_score(transport, latent):
    """Vaitl (2024) Prop.3.2, layerwise; no coupling-specific O(d) claim."""
    values, score = latent, -latent
    logdet = tf.zeros(tf.shape(latent)[0], latent.dtype)
    for layer in _score_layers(transport):
        values, ld, jacobian, ld_score = value_jacobian_logdet_score(layer, values)
        rhs = tf.stop_gradient(score-ld_score)[..., None]
        matrix = tf.stop_gradient(jacobian)
        if getattr(layer, "autoregressive", False):
            score = tf.linalg.triangular_solve(matrix, rhs, lower=True, adjoint=True)[..., 0]
        else:
            score = tf.linalg.solve(tf.linalg.matrix_transpose(matrix), rhs)[..., 0]
        logdet += ld
    return values, logdet, tf.stop_gradient(score)


def reverse_kl_evaluate(transport, target_value_score, latent, *, estimator, variables=None,
                        target_dtype=None):
    variables = tuple(transport.trainable_variables if variables is None else variables)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        if estimator == "path":
            physical, logdet, proposal_score = forward_proposal_score(transport, latent)
        elif estimator == "standard":
            physical, logdet = transport.forward_and_logdet(latent)
            proposal_score = tf.zeros_like(physical)
        else:
            raise ValueError("estimator must be standard or path")
        target_points = tf.stop_gradient(physical)
        if target_dtype is not None:
            target_points = tf.cast(target_points, target_dtype)
        values, score, status = target_value_score(target_points)
        values = tf.ensure_shape(tf.convert_to_tensor(values), latent.shape[:1])
        score = tf.ensure_shape(tf.convert_to_tensor(score), latent.shape)
        if values.dtype not in (tf.float32, tf.float64) or score.dtype != values.dtype:
            raise ValueError("target values and scores must share float32 or float64 dtype")
        status = tf.ensure_shape(tf.convert_to_tensor(status, tf.bool), latent.shape[:1])
        # RKL up to the parameter-independent base log-density constant.
        actual_loss = tf.reduce_mean(-values-tf.cast(logdet, values.dtype))
        # Keep target values/scores in their own precision. Casting this carrier
        # backpropagates to the flow dtype, including FP32 parameter gradients.
        physical_target = tf.cast(physical, values.dtype)
        if estimator == "path":
            carrier = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(
                tf.cast(proposal_score, score.dtype)-score)*physical_target, axis=-1))
        else:
            attached = tf.stop_gradient(values)+tf.reduce_sum(
                (physical_target-tf.stop_gradient(physical_target))*tf.stop_gradient(score), axis=-1)
            carrier = tf.reduce_mean(-attached-tf.cast(logdet, values.dtype))
    gradients = tape.gradient(carrier, variables)
    if any(g is None for g in gradients):
        raise ValueError("missing transport parameter gradient")
    valid = tf.reduce_all(tf.stack((tf.reduce_all(status), *(
        tf.reduce_all(tf.math.is_finite(v)) for v in
        (latent, physical, values, score, logdet, proposal_score, actual_loss, carrier, *gradients)))))
    return {"loss": actual_loss, "gradients": tuple(gradients), "valid": valid,
            "proposal_score": proposal_score}


class AutoregressiveStage:
    """Shared configured IAF or Huang conditional DSF layer."""
    autoregressive = True

    def __init__(self, config, index, *, trainable=True):
        self.config = config
        self.dtype = tf.as_dtype(config.dtype)
        self.dimension = self.parameter_dim = config.dimension
        self.hidden_layers = config.hidden_layers
        self.activation = config.activation
        self.s_max = config.conditional_scale_cap
        self.scale_transform = config.scale_transform
        self.kind = config.kind
        if self.kind == "naf_dsf" and config.naf_conditioner == "author_cmade":
            self._initialize_author_conditioner(config, index, trainable)
            return
        count = 2 if self.kind == "iaf" else 3*config.mixture_components
        self.masks = (hoffman_masks(self.dimension, self.hidden_layers, dtype=self.dtype)
                      if config.mask_policy == "hoffman_block_masks_v1" else
                      dense_masks(self.dimension, self.hidden_layers, outputs_per_dimension=count, dtype=self.dtype))
        sizes = (self.dimension, *self.hidden_layers, count*self.dimension)
        root = tf.random.experimental.stateless_fold_in(tf.constant(config.seed, tf.int32), index)
        weights, biases = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            seed = tf.random.experimental.stateless_fold_in(root, i)
            # Glorot uniform for hidden layers (documented implementation choice);
            # author NAF's small nonzero final weights break component symmetry.
            radius = (config.final_weight_scale if self.kind == "naf_dsf" or i == len(sizes)-2
                      else math.sqrt(6./(n_in+n_out)))
            w = tf.random.stateless_uniform((n_in, n_out), seed, minval=-radius, maxval=radius, dtype=self.dtype)
            if self.kind == "iaf" and config.iaf_initializer == "hoffman_variance_scaling":
                # MakeIAFBijectorFn -> L2HMCInitializer(.01): fan-in variance
                # scale 2*.01. TF variance-scaling corrects truncation at 2 SD.
                stddev = math.sqrt(config.iaf_variance_scale/n_in)/.87962566103423978
                w = tf.random.stateless_truncated_normal((n_in, n_out), seed, dtype=self.dtype)*tf.constant(stddev, self.dtype)
            b = tf.zeros((n_out,), self.dtype)
            if i == len(sizes)-2 and self.kind == "naf_dsf":
                # softplus(raw)+slope_floor=1 at the zero-conditioner origin.
                n = config.mixture_components*self.dimension
                base = math.log(math.expm1(1.-config.slope_floor))
                b = tf.concat((tf.fill([n], tf.constant(base, self.dtype)), tf.zeros([2*n], self.dtype)), axis=0)
            if trainable:
                w, b = tf.Variable(w, name=f"neutra_{index}_weight_{i}"), tf.Variable(b, name=f"neutra_{index}_bias_{i}")
            weights.append(w)
            biases.append(b)
        self.weights, self.biases = tuple(weights), tuple(biases)
        self.trainable_variables = tuple(v for pair in zip(weights, biases) for v in pair) if trainable else ()

    @property
    def extra_parameters(self):
        if self.kind != "naf_dsf" or self.config.naf_conditioner != "author_cmade":
            return {}
        return {name: getattr(self, name) for name in (
            "context_scale_weights", "context_scale_biases", "context_bias_weights", "projection")}

    def _initialize_author_conditioner(self, config, index, trainable):
        """Huang's IAF_DSF + cMADE/CWNlinear with constant context [1].

        Constant context is the unconditional target specialization. Retain both
        context weights and biases, the final conditioner weight normalization,
        and the shared 1x1 projection, rather than folding them into a new MLP.
        Hidden degree order is frozen to a balanced valid author rank assignment.
        """
        self.features = 3*(config.hidden_layers[-1]//self.dimension)
        self.masks = dense_masks(self.dimension, self.hidden_layers, outputs_per_dimension=self.features, dtype=self.dtype)
        sizes = (self.dimension, *self.hidden_layers, self.features*self.dimension)
        root = tf.random.experimental.stateless_fold_in(tf.constant(config.seed, tf.int32), index)
        def random(shape, key, *, normal=False, radius=1.):
            seed = tf.random.experimental.stateless_fold_in(root, key)
            return (tf.random.stateless_normal(shape, seed, dtype=self.dtype)*tf.constant(radius, self.dtype)
                    if normal else tf.random.stateless_uniform(shape, seed, minval=-radius, maxval=radius, dtype=self.dtype))
        def parameter(value, name):
            return tf.Variable(value, name=f"neutra_{index}_{name}") if trainable else value
        weights, biases, scale_weights, scale_biases, bias_weights = [], [], [], [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            weights.append(parameter(random([n_in, n_out], 10*i, normal=True, radius=.001), f"direction_{i}"))
            # nn.Linear(1,out): bias uniform +/-1; author overwrites context
            # weights with Normal(0,.001), preserving the original biases.
            biases.append(parameter(random([n_out], 10*i+1), f"context_bias_bias_{i}"))
            scale_weights.append(parameter(random([n_out], 10*i+2, normal=True, radius=.001), f"context_scale_weight_{i}"))
            scale_biases.append(parameter(random([n_out], 10*i+3), f"context_scale_bias_{i}"))
            bias_weights.append(parameter(random([n_out], 10*i+4, normal=True, radius=.001), f"context_bias_weight_{i}"))
        self.weights, self.biases = tuple(weights), tuple(biases)
        self.context_scale_weights, self.context_scale_biases = tuple(scale_weights), tuple(scale_biases)
        self.context_bias_weights = tuple(bias_weights)
        k = config.mixture_components
        pweight = random([self.features, 3*k], 10001, radius=config.final_weight_scale)
        pbias = tf.concat((tf.fill([k], tf.constant(math.log(math.expm1(1.-config.slope_floor)), self.dtype)),
                           tf.zeros([2*k], self.dtype)), axis=0)
        self.projection = (parameter(pweight, "projection_weight"), parameter(pbias, "projection_bias"))
        variables = tuple(v for pair in zip(weights, biases) for v in pair)
        variables += tuple(v for entries in self.extra_parameters.values() for v in entries)
        self.trainable_variables = variables if trainable else ()

    def _network(self, values):
        if self.kind != "iaf":
            raise ValueError("affine scale diagnostics do not apply to NAF")
        return iaf_network(self, values)

    def pseudo_parameters(self, values):
        if self.config.naf_conditioner == "author_cmade":
            hidden = values
            for i, (direction, bias, mask) in enumerate(zip(self.weights, self.biases, self.masks)):
                # Source CWNlinear normalizes only the final conditioner layer,
                # before applying its mask. No normalization epsilon is added.
                weight = direction
                if i == len(self.weights)-1:
                    weight = weight/tf.sqrt(tf.reduce_sum(tf.square(weight), axis=0, keepdims=True))
                scale = self.context_scale_weights[i]+self.context_scale_biases[i]
                hidden = scale*tf.matmul(hidden, weight*mask)+bias+self.context_bias_weights[i]
                if i+1 < len(self.weights):
                    hidden = activation(hidden, self.activation)
            features = tf.transpose(tf.reshape(hidden, [tf.shape(values)[0], self.features, self.dimension]), (0, 2, 1))
            raw = tf.matmul(features, self.projection[0])+self.projection[1]
            raw = tf.reshape(raw, [tf.shape(values)[0], self.dimension, 3, self.config.mixture_components])
        else:
            raw, _ = masked_network(self, values)
            # Grouped masks: [parameter, sigmoid component, coordinate].
            raw = tf.reshape(raw, [tf.shape(values)[0], 3, self.config.mixture_components, self.dimension])
            raw = tf.transpose(raw, (0, 3, 1, 2))
        slope = tf.nn.softplus(raw[:, :, 0, :])+tf.constant(self.config.slope_floor, self.dtype)
        return tf.math.log(slope), raw[:, :, 1, :], raw[:, :, 2, :]

    def forward_and_logdet(self, values):
        if self.kind == "iaf":
            return iaf_forward(self, values)
        output, ld, _ = sigmoid_mixture(values, *self.pseudo_parameters(values))
        return output, tf.reduce_sum(ld, axis=-1)

    def pullback_score(self, values, score):
        if self.kind == "iaf":
            return iaf_pullback(self, values, score)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(values)
            output, _ = self.forward_and_logdet(values)
        return tape.gradient(output, values, output_gradients=score)

    def logdet_score(self, values):
        if self.kind == "iaf":
            return iaf_logdet_score(self, values)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(values)
            _, ld = self.forward_and_logdet(values)
        return tape.gradient(ld, values, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    def _inverse_value(self, output):
        def solve(i, x, valid):
            slopes, offsets, logits = self.pseudo_parameters(x)
            solved, _, ok, _ = sigmoid_inverse(output[:, i], slopes[:, i], offsets[:, i], logits[:, i],
                atol=self.config.inverse_atol, rtol=self.config.inverse_rtol,
                max_iterations=self.config.inverse_max_iterations)
            x += (solved-x[:, i])[:, None]*tf.one_hot(i, self.dimension, dtype=self.dtype)
            return i+1, x, valid & ok
        _, x, valid = tf.while_loop(lambda i, x, valid: i < self.dimension, solve,
            (tf.constant(0), tf.zeros_like(output), tf.reduce_all(tf.math.is_finite(output), axis=-1)),
            parallel_iterations=1, maximum_iterations=self.dimension)
        return tf.where(valid[:, None], x, tf.constant(float("nan"), self.dtype))

    def inverse(self, output):
        if self.kind == "iaf":
            return iaf_inverse(self, output)

        @tf.custom_gradient
        def implicit_inverse(y):
            x = tf.stop_gradient(self._inverse_value(y))
            def gradient(dx, variables=None):
                # dy = J dx + (dT/dphi) dphi, hence dx/dphi=-J^-1 dT/dphi.
                _, _, jacobian, _ = value_jacobian_logdet_score(self, x)
                dy = tf.linalg.triangular_solve(jacobian, dx[..., None], lower=True, adjoint=True)[..., 0]
                if variables is None:
                    return dy
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(variables)
                    forward, _ = self.forward_and_logdet(x)
                dv = tape.gradient(forward, variables, output_gradients=-dy,
                                   unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return dy, dv
            return x, gradient
        return implicit_inverse(output)

    def inverse_and_forward_logdet(self, output):
        x = self.inverse(output)
        return x, self.forward_and_logdet(x)[1]

    def scale_diagnostics(self, values):
        if self.kind == "iaf":
            scale, _, derivative, _, _ = iaf_parameters(self, values)
            return scale, derivative
        _, ld, _ = sigmoid_mixture(values, *self.pseudo_parameters(values))
        # No conditional tanh cap in DSF. Report diagonal log derivative only.
        return ld, tf.ones_like(ld)


class Permutation:
    def __init__(self, dimension, policy="full_reverse"):
        self.parameter_dim = dimension
        self.indices = (tuple(reversed(range(dimension))) if policy == "full_reverse"
                        else (0, *reversed(range(1, dimension))))
        self.trainable_variables = ()

    def forward_and_logdet(self, values):
        return tf.gather(values, self.indices, axis=-1), tf.zeros(tf.shape(values)[:-1], values.dtype)

    def inverse(self, values):
        return tf.gather(values, self.indices, axis=-1)

    def pullback_score(self, values, score):
        return self.inverse(score)

    def logdet_score(self, values):
        return tf.zeros_like(values)


class AffineLayer:
    """Configured affine view used when traversing wrapped legacy maps."""
    def __init__(self, shift, scale=None, *, matrix=None):
        self.shift, self.scale, self.matrix = shift, scale, matrix

    def forward_and_logdet(self, values):
        return affine_forward(values, self.shift, self.scale, matrix=self.matrix)

    def inverse(self, values):
        return affine_inverse(values, self.shift, self.scale, matrix=self.matrix)[0]

    def pullback_score(self, values, score):
        return affine_pullback(score, self.scale, matrix=self.matrix)

    def logdet_score(self, values):
        return tf.zeros_like(values)
