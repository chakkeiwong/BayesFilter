"""Analytic diagnostic quantities; no tuning or sampler-admission authority.

The exact/partial funnel charts describe V~N(0,9), X|V~N(0,exp(V) I).
Reference intervals use the existing lugsail estimator under its stationary
CLT assumptions. They are approximate, fixed-horizon diagnostics, not valid
confidence sequences for repeatedly stopped chains.
"""
from dataclasses import asdict, dataclass
import math
from statistics import NormalDist

import tensorflow as tf

from .designs import digest
from .targets import ValidationTarget


QUANTITIES_ID = "d4r3_analytic_model_quantities_v2"


class ExactWhitenedFunnelTarget(ValidationTarget):
    """Diagnostic exact chart of the same scale-three partial-funnel model."""

    def __init__(self, *, jit_compile=True):
        super().__init__("funnel_noncentered", {"scale": 1.}, jit_compile=jit_compile)

    def to_model(self, q):
        v = 3.*q[..., :1]
        return tf.concat([v, tf.exp(v/2.)*q[..., 1:]], -1)

    def adapter_signature(self):
        return digest({"analytic_exact_funnel": "v1", "scale": 3.,
                       "model_chart": "v=3*z0;x=exp(v/2)*zchild"})


@dataclass(frozen=True)
class AnalyticQuantity:
    name: str
    expectation: float | None
    coordinates: str
    monitor: bool = True
    binary: bool = False

    @property
    def screen(self):
        return self.monitor and self.expectation is not None


@dataclass(frozen=True)
class AnalyticQuantities:
    target: str
    dimension: int
    specifications: tuple[AnalyticQuantity, ...]
    values: object
    posterior_quantities: object

    @property
    def parameter_names(self):
        return tuple(f"q{i}" for i in range(self.dimension))

    @property
    def precision_names(self):
        return tuple(s.name for s in self.specifications if s.monitor)

    @property
    def binary_names(self):
        return tuple(s.name for s in self.specifications if s.monitor and s.binary)

    def payload(self):
        return dict(quantities_id=QUANTITIES_ID, target=self.target,
                    quantities=[dict(**asdict(s), screen=s.screen) for s in self.specifications],
                    tuning_authority=False, posterior_correctness_authority=False)


def analytic_quantities(target, adapter, *, use_xla=True):
    """Freeze quantity identities before sampling, with exact expectations.

    State-space adapters retain their coordinate/square monitoring but have
    no analytic posterior reference here. Full-model references are separate.
    """
    from bayesfilter.testing.acceptance_uncertainty_models import ResidualFunnelTarget

    d = adapter.parameter_dim
    parameters = getattr(adapter, "parameters", {})
    funnel = target in {"funnel_noncentered", "residual_funnel"}
    if target == "funnel_noncentered" and not isinstance(adapter, ExactWhitenedFunnelTarget):
        raise ValueError("exact funnel reference requires the declared scale-three chart")
    if target == "residual_funnel" and not isinstance(adapter, ResidualFunnelTarget):
        raise ValueError("partial funnel reference requires the declared residual chart")
    mean, second = [None]*d, [None]*d
    square_monitor = True
    if target in {"gaussian", "funnel_noncentered", "residual_funnel", "student_t"}:
        mean = [0.]*d
        second = [1.]*d
        if target == "gaussian":
            second = [float(parameters.get("scale", 1.))**2]*d
        elif target == "student_t":
            df = float(parameters.get("df", 5.))
            second = [df/(df-2)]*d
            square_monitor = df > 4  # Var(q^2) requires a finite fourth moment.
        elif target == "residual_funnel":
            second = [1.] + [None]*(d-1)
    elif target == "rotated_gaussian":
        angle, condition = parameters.get("angle", .6), parameters.get("condition", 9.)
        c, s = math.cos(angle), math.sin(angle)
        mean, second = [0., 0.], [c*c+condition*s*s, s*s+condition*c*c]
    elif target == "mixture":
        weight, separation = parameters.get("weight", .3), parameters.get("separation", 5.)
        mean, second = [(1-2*weight)*separation, 0.], [1+separation**2, 1.]
    specs = [AnalyticQuantity(f"q{i}", mean[i], "adapter") for i in range(d)]
    specs += [AnalyticQuantity(f"q{i}_square", second[i], "adapter", square_monitor) for i in range(d)]
    if target == "student_t":
        tail = float(tf.math.betainc(tf.constant(df/2, tf.float64),
                                    tf.constant(.5, tf.float64), tf.constant(df/(df+4), tf.float64)))
        specs += [AnalyticQuantity(f"q{i}_abs_gt_2", tail, "adapter", binary=True) for i in range(d)]
    if target == "mixture":
        left = weight*NormalDist().cdf(separation) + (1-weight)*NormalDist().cdf(-separation)
        specs.append(AnalyticQuantity("model_left_mass", left, "model", binary=True))
    if funnel:
        specs += [AnalyticQuantity("model_v", 0., "model"),
                  AnalyticQuantity("model_v_square", 9., "model")]
        for i in range(1, d):
            # All moments exist, but the large fourth moment makes a short
            # child-square MCSE unreliable. Preserve these descriptively.
            specs += [AnalyticQuantity(f"model_x{i}", 0., "model", monitor=False),
                      AnalyticQuantity(f"model_x{i}_square", math.exp(4.5), "model", monitor=False)]
        neck, tail = NormalDist().cdf(-1.), math.erfc(math.sqrt(2.))
        specs += [AnalyticQuantity("model_neck_mass", neck, "model", binary=True),
                  AnalyticQuantity("model_flare_mass", neck, "model", binary=True)]
        for i in range(1, d):
            specs += [AnalyticQuantity(f"model_x{i}_conditional_tail", tail, "model", binary=True),
                      AnalyticQuantity(f"model_x{i}_neck_tail", neck*tail, "model", binary=True)]
    specs = tuple(specs)
    signature = [tf.TensorSpec([None, None, d], tf.float64)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=use_xla)
    def values(q):
        result = {f"q{i}": q[..., i] for i in range(d)}
        result.update({f"q{i}_square": tf.square(q[..., i]) for i in range(d)})
        if target == "student_t":
            result.update({f"q{i}_abs_gt_2": tf.cast(tf.abs(q[..., i]) > 2., tf.float64) for i in range(d)})
        if target == "mixture":
            result["model_left_mass"] = tf.cast(q[..., 0] < 0., tf.float64)
        if funnel:
            model = adapter.to_model(q)
            v = model[..., 0]
            result.update(model_v=v, model_v_square=tf.square(v),
                          model_neck_mass=tf.cast(v < -3., tf.float64),
                          model_flare_mass=tf.cast(v > 3., tf.float64))
            for i in range(1, d):
                x = model[..., i]
                event = tf.abs(x) > 2.*tf.exp(v/2.)
                result.update({f"model_x{i}": x, f"model_x{i}_square": tf.square(x),
                    f"model_x{i}_conditional_tail": tf.cast(event, tf.float64),
                    f"model_x{i}_neck_tail": tf.cast(event & (v < -3.), tf.float64)})
        return result

    @tf.function(input_signature=signature, autograph=False, jit_compile=use_xla)
    def posterior_quantities(q):
        result = values(q)
        return {s.name: result[s.name] for s in specs if s.monitor and s.name not in {f"q{i}" for i in range(d)}}

    return AnalyticQuantities(target, d, specs, values, posterior_quantities)


def analytic_reference_report(samples, quantities, *, family_alpha=.05, use_xla=True):
    """Post-run reference diagnostic. Inconclusive quantities never pass.

    Keep the declared family size even when some MCSEs are unavailable.
    Discrepancy and inconclusiveness do not change tuning membership. The
    sample window may have been stopped; this diagnostic grants no coverage
    guarantee for that stopping rule.
    """
    from bayesfilter.inference.hmc_precision import mean_precision

    if not math.isfinite(family_alpha) or not 0 < family_alpha < 1:
        raise ValueError("family_alpha must be strictly between zero and one")
    specs = quantities.specifications
    screened = tuple(s for s in specs if s.screen)
    result = dict(schema="bayesfilter.mams_analytic_reference.v1", **quantities.payload(),
                  family_alpha=family_alpha, family_size=len(screened), rows=[],
                  accuracy_established=False, ranking_supported=False,
                  uncertainty="approximate fixed-horizon lugsail/normal Bonferroni; no stopped-chain coverage guarantee")
    if samples is None:
        return dict(result, status="inconclusive", reason="no_retained_draws")
    q = tf.convert_to_tensor(samples, tf.float64)
    if q.shape.rank != 3 or not q.shape.is_fully_defined() or q.shape[-1] != quantities.dimension:
        raise ValueError("analytic reference requires fixed [draw,chain,adapter_coordinate]")
    if q.shape[0] < 4 or q.shape[1] < 2:
        return dict(result, status="inconclusive", reason="insufficient_draws_or_chains")
    values = quantities.values(q)
    if not all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in values.values()):
        return dict(result, status="invalid", reason="nonfinite_quantities")
    estimates = {name: float(tf.reduce_mean(value)) for name, value in values.items()}
    result["descriptive_estimates"] = estimates
    if not screened:
        return dict(result, status="reference_unavailable", reason="no_analytic_reference_for_this_target")
    precision = mean_precision(tf.stack([values[s.name] for s in screened], axis=-1),
                               method="lugsail", jit_compile=use_xla)
    critical = NormalDist().inv_cdf(1-family_alpha/(2*len(screened)))
    result["critical_value"] = critical
    for index, spec in enumerate(screened):
        valid = bool(precision["valid"][index])
        estimate = estimates[spec.name]
        error = estimate-spec.expectation
        mcse = float(precision["mcse"][index]) if valid else None
        interval = [estimate-critical*mcse, estimate+critical*mcse] if valid else None
        finding = ("no_discrepancy_detected" if abs(error) <= critical*mcse else "discrepancy_detected") if valid else "inconclusive"
        result["rows"].append(dict(name=spec.name, expectation=spec.expectation, estimate=estimate,
            error=error, mcse=mcse, interval=interval, finding=finding))
    findings = {row["finding"] for row in result["rows"]}
    result["status"] = ("discrepancy_detected" if "discrepancy_detected" in findings else
                        "inconclusive" if "inconclusive" in findings else "no_discrepancy_detected")
    return result
