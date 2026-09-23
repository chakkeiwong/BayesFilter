"""Independent diagnostic for one predetermined output per normal-conjugate fit.

Under the correct conditional-output null, independent standardized endpoints
are iid N(0,1). Two fixed tests at alpha/2 screen location and second moment.
This tests neither stopped-functional coverage nor achieved full-HMC power.
SciPy is an independent diagnostic authority, never a tuning decision path.
"""
import math

from scipy import stats

OUTPUT_RULE = "first_verified_id_last_retained_draw_chain0.v1"


def normal_reference(data, *, tau=2., sigma=1.):
    if (not data or any(type(v) not in (int, float) or not math.isfinite(v) for v in data)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (tau, sigma))):
        raise ValueError("finite data and positive finite prior/noise scales required")
    variance = 1. / (1. / tau**2 + len(data) / sigma**2)
    return variance * math.fsum(data) / sigma**2, math.sqrt(variance)


def endpoint_from_pipeline(fit_id, pipeline, *, tau=2., sigma=1.):
    """Read the frozen selection; capped or unqualified fits remain unavailable."""
    from ..storage import read_tensor
    mean, sd = normal_reference(pipeline["data"], tau=tau, sigma=sigma)
    record = {"fit_id": fit_id, "output_rule": OUTPUT_RULE, "reference_mean": mean,
              "reference_sd": sd, "status": "unavailable", "z": None, "stream": None}
    selected = pipeline["selection"]
    ids = selected["candidate_ids"]
    verified = pipeline["verified_candidate_ids"]
    if (selected["rule"] != "first_verified" or selected["scope"] != "selected"
            or len(ids) != 1 or not verified or ids[0] != sorted(verified)[0]):
        raise ValueError("endpoint requires predetermined first-verified member rule")
    members = [m for m in pipeline["members"] if m["candidate_id"] == ids[0]]
    if len(members) != 1:
        raise ValueError("missing or duplicated selected member")
    member = members[0]
    posterior = member.get("posterior", {})
    qualified = (pipeline["completion"] == "complete" and member["status"] == "assessed"
        and posterior.get("passed") is True and posterior.get("warmup_cap_hit") is False
        and posterior.get("retained_cap_hit") is False and not posterior.get("hard_vetoes", [])
        and member.get("warmup_exclusion_matches") is True)
    record["reason"] = "qualified" if qualified else "missing_capped_or_unqualified_posterior"
    if not qualified:
        return record
    values = read_tensor(member["draws_path"])
    if len(values.shape) != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] != 1:
        raise ValueError("normal endpoint requires nonempty [draw,chain,1] model samples")
    x = float(values[-1, 0, 0])
    if not math.isfinite(x):
        raise ValueError("nonfinite endpoint")
    record.update(status="qualified", value=x, z=(x-mean)/sd,
                  stream=list(posterior["config"]["retained_seed"]))
    return record


def assess_endpoints(records, planned_fit_ids, *, alpha=.05):
    """No available-case denominator and no unavailable result counted as detection."""
    planned = tuple(planned_fit_ids)
    if not planned or any(not isinstance(i, str) or not i for i in planned) or len(set(planned)) != len(planned):
        raise ValueError("nonempty unique planned fit IDs required")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rows = {}
    streams = set()
    for record in records:
        fit_id = record["fit_id"]
        if fit_id not in planned or fit_id in rows or record["output_rule"] != OUTPUT_RULE:
            raise ValueError("unknown/duplicate fit or changed output rule")
        if record["status"] not in {"qualified", "unavailable"}:
            raise ValueError("unknown endpoint status")
        if record["status"] == "qualified":
            if type(record["z"]) not in (int, float) or not math.isfinite(record["z"]):
                raise ValueError("qualified endpoint must be finite")
            stream = tuple(record["stream"] or ())
            if len(stream) != 2 or any(type(x) is not int for x in stream) or stream in streams:
                raise ValueError("independent fit streams required")
            streams.add(stream)
        rows[fit_id] = record
    available = [r for r in rows.values() if r["status"] == "qualified"]
    report = {"planned": len(planned), "available": len(available),
              "unavailable": len(planned)-len(available), "output_rule": OUTPUT_RULE,
              "alpha": alpha, "component_alpha": alpha/2, "status": "incomplete",
              "reject": None, "tests": None,
              "interpretation": "diagnostic only; correct-output null is an assumption being tested"}
    if len(available) != len(planned):
        return report
    n = len(available)
    z = [r["z"] for r in available]
    mean_stat = math.fsum(z) / math.sqrt(n)
    square_stat = math.fsum(v*v for v in z)
    mean_p = float(2 * stats.norm.sf(abs(mean_stat)))
    square_p = float(min(1., 2 * min(stats.chi2.cdf(square_stat, n), stats.chi2.sf(square_stat, n))))
    report.update(status="complete", reject=bool(min(mean_p, square_p) <= alpha/2),
                  tests={"mean": {"statistic": mean_stat, "p_value": mean_p},
                         "second_moment": {"statistic": square_stat, "p_value": square_p}})
    return report
