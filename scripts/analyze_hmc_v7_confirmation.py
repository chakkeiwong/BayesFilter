"""Diagnostic full-procedure price and fixed-denominator delivery analysis.

No TensorFlow initialization, sampling, launch authority or release authority.
Run with ``python -m scripts.analyze_hmc_v7_confirmation --help``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from scripts.run_hmc_v7_release_prices import full_search_delivered


FAMILIES = ("lgssm_qr", "nonlinear", "funnel_residual")
PRIMARY_L = (3, 5, 9, 13, 18, 25)


def _full_release_delivery(result, family):
    """A completed single-pair diagnostic is not the declared broad procedure."""
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration

    search = result.get("search", {})
    seed = result.get("sampling_streams", ())
    if (not isinstance(seed, (list, tuple)) or len(seed) != 2
            or any(type(v) is not int or not 0 <= v < 2**31 for v in seed)):
        return False
    if result.get("classification") not in {"development", "regression", "confirmation", "scheduled_stress"}:
        return False
    cap = search.get("max_wall_time_seconds")
    if (isinstance(cap, bool) or not isinstance(cap, (int, float))
            or not math.isfinite(cap) or not 0 < cap <= 3600):
        return False
    expected = full_search_configuration(family, seed=seed, wall_seconds=cap,
                                        classification=result["classification"])["search"]
    # Canonical serialized policies also distinguish bool from numeric fields.
    # Actual results have already passed the controller's payload normalization.
    same_search = (json.dumps(search, sort_keys=True, allow_nan=False)
                   == json.dumps(expected, sort_keys=True, allow_nan=False))
    return (full_search_delivered(result)
            and result.get("schema") == "bayesfilter.replicated_acceptance_model_result.v1"
            and result.get("case_id") == str(result.get("classification")) + "-" + family
            and same_search
            and set(PRIMARY_L) <= {row["leapfrog_steps"] for row in result.get("candidates", [])})


def _probability(value):
    if isinstance(value, bool):
        raise ValueError("probability must be a number")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError("probability must be a finite number") from None
    if not 0 <= result <= 1:
        raise ValueError("probability is outside [0, 1]")
    return result


def binomial_upper_tail(successes, total, probability):
    """Exact rational P_p(X >= successes) for integer binomial counts."""
    if (type(total) is not int or total < 1 or type(successes) is not int
            or not 0 <= successes <= total):
        raise ValueError("invalid binomial counts")
    p = _probability(probability)
    a, b = p.numerator, p.denominator
    numerator = sum(math.comb(total, k) * a**k * (b-a)**(total-k)
                    for k in range(successes, total+1))
    return Fraction(numerator, b**total)


def one_sided_lower_bound(successes, total, alpha):
    """Bracket the binomial-tail root exactly, then round the display down."""
    alpha = _probability(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly inside (0, 1)")
    binomial_upper_tail(successes, total, 0)  # Validate even the zero-success case.
    if successes == 0:
        return 0.0
    low, high = Fraction(0), Fraction(1)
    for _ in range(64):
        middle = (low + high) / 2
        if binomial_upper_tail(successes, total, middle) < alpha:
            low = middle
        else:
            high = middle
    return max(0.0, math.nextafter(float(low), 0.0))


def delivery_report(*, slots, outcomes, source_manifest_sha256,
                    familywise_alpha="0.05", required_delivery="0.80"):
    """Use every planned slot; success flags supplied by callers are ignored.

    Each outcome carries the model's complete structured result, rather than
    a member count or an unverified delivery Boolean. This reporting function
    does not authenticate files or replace the numerical artifact validators.
    """
    alpha = _probability(familywise_alpha)
    requirement = _probability(required_delivery)
    if not 0 < alpha < 1 or not 0 < requirement < 1:
        raise ValueError("alpha and required delivery must be inside (0, 1)")
    if not isinstance(source_manifest_sha256, str) or not source_manifest_sha256:
        raise ValueError("a frozen source identity is required")
    planned, seeds = {}, set()
    for slot in slots:
        sid, family = slot["slot_id"], slot["family"]
        seed = tuple(slot["seed"])
        if (not isinstance(sid, str) or not sid or sid in planned or family not in FAMILIES
                or len(seed) != 2 or any(type(v) is not int or not 0 <= v < 2**31 for v in seed)
                or seed in seeds):
            raise ValueError("duplicate or invalid planned slot/family/seed")
        planned[sid] = slot
        seeds.add(seed)
    counts = Counter(slot["family"] for slot in planned.values())
    if set(counts) != set(FAMILIES):
        raise ValueError("all three declared families need a planned denominator")
    results = {}
    dispositions = Counter()
    successes = Counter()
    for outcome in outcomes:
        sid = outcome["slot_id"]
        if sid not in planned or sid in results:
            raise ValueError("unexpected or duplicate outcome slot")
        slot = planned[sid]
        if (outcome["family"] != slot["family"] or tuple(outcome["seed"]) != tuple(slot["seed"])
                or outcome["source_manifest_sha256"] != source_manifest_sha256):
            raise ValueError("outcome differs from the frozen slot/source")
        code = outcome["exit_code"]
        if type(code) is not int or code not in (0, 2, 3, 124):
            raise ValueError("unclassified execution failure invalidates the reporting input")
        model = outcome.get("model_result", {})
        model_seed = model.get("sampling_streams", ())
        if model and (not isinstance(model_seed, (list, tuple))
                      or tuple(model_seed) != tuple(slot["seed"])
                      or any(type(v) is not int for v in model_seed)):
            raise ValueError("model result seed differs from the frozen slot")
        successful = code == 0 and _full_release_delivery(model, slot["family"])
        label = ("complete_delivery" if successful else "resource_deferred" if code == 3
                 else "timeout" if code == 124 else "no_complete_delivery")
        results[sid] = successful
        dispositions[label] += 1
        successes[slot["family"]] += successful
    missing = sorted(set(planned) - set(results))
    dispositions["missing"] = len(missing)
    per_family = alpha / len(FAMILIES)
    rows = []
    for family in FAMILIES:
        n, s = counts[family], successes[family]
        tail = binomial_upper_tail(s, n, requirement)
        rows.append(dict(family=family, planned=n, successes=s, unsuccessful=n-s,
                         one_sided_lower_bound=one_sided_lower_bound(s, n, per_family),
                         exact_tail_at_requirement=str(tail), alpha=str(per_family),
                         lower_bound_exceeds_requirement=tail < per_family))
    return dict(schema="bayesfilter.hmc_v7_confirmation_analysis.v1", families=rows,
                original_denominator=len(planned), missing_slots=missing,
                dispositions=dict(dispositions), all_slots_reported=not missing,
                delivery_criterion_passed=all(row["lower_bound_exceeds_requirement"] for row in rows),
                familywise_alpha=str(alpha), required_delivery=str(requirement),
                source_manifest_sha256=source_manifest_sha256, release_ready=False,
                default_promotion=False,
                interpretation="Conditional complete-work numerical delivery over independent seeds; "
                               "missing/resource outcomes counted unsuccessful, no deadline guarantee.")


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be a finite nonnegative number")
    return float(value)


def price_report(price_roots, *, replications_per_family, available_gpu_seconds):
    """Read full model results; a partial or one-family price cannot fund the design."""
    if type(replications_per_family) is not int or replications_per_family < 1:
        raise ValueError("positive predeclared replication count required")
    budget = _seconds(available_gpu_seconds)
    complete, issues, seen = {}, [], set()
    sources, devices, inputs = set(), set(), []
    for path in price_roots:
        path = Path(path).resolve()
        raw = path.read_bytes()
        outer = json.loads(raw)
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        if outer.get("schema") != "bayesfilter.hmc_v7_full_search_prices.v1":
            raise ValueError("unsupported price schema")
        cases, attempts = outer["cases"], outer["attempts"]
        if (not cases or len(set(cases)) != len(cases) or set(cases) - set(FAMILIES)
                or any(case in seen for case in cases)):
            raise ValueError("duplicate or unexpected priced family")
        seen.update(cases)
        if (len({row["case"] for row in attempts}) != len(attempts)
                or any(row["case"] not in cases for row in attempts)):
            raise ValueError("duplicate or unexpected price attempt")
        enclosing = _seconds(outer["wall_seconds"])
        summed = math.fsum(_seconds(row["wall_seconds"]) for row in attempts)
        if summed > enclosing:
            raise ValueError("nested prices exceed enclosing charge")
        if (outer.get("status") != "complete" or outer.get("unstarted_cases")
                or {row["case"] for row in attempts} != set(cases)):
            issues.append(dict(cases=cases, reason="incomplete_price"))
            continue
        for key in ("source_manifest_sha256", "gpu_uuid"):
            if not isinstance(outer.get(key), str) or not outer[key]:
                raise ValueError("price source/device identity is missing")
        sources.add(outer["source_manifest_sha256"])
        devices.add(outer["gpu_uuid"])
        overhead = (enclosing - summed) / len(attempts)
        for row in attempts:
            model_path = path.parent / row["case"] / "model" / "result.json"
            model = json.loads(model_path.read_text()) if model_path.exists() else {}
            if (type(row["exit_code"]) is not int or row["exit_code"] != 0
                    or not _full_release_delivery(model, row["case"])):
                issues.append(dict(cases=[row["case"]], reason="complete_numerical_result_missing"))
                continue
            complete[row["case"]] = _seconds(row["wall_seconds"]) + overhead
    if len(sources) > 1 or len(devices) > 1:
        raise ValueError("price source/device differs across families")
    missing = sorted(set(FAMILIES) - set(complete))
    forecast = (replications_per_family * math.fsum(complete.values())
                if not missing and not issues else None)
    return dict(schema="bayesfilter.hmc_v7_confirmation_price_analysis.v1",
                inputs=inputs, complete_case_seconds=complete, missing_cases=missing, issues=issues,
                replications_per_family=replications_per_family, available_gpu_seconds=budget,
                point_forecast_seconds=forecast,
                point_forecast_fits_budget=forecast is not None and forecast <= budget,
                source_manifest_sha256=next(iter(sources), None), gpu_uuid=next(iter(devices), None),
                confirmation_authorized=False, release_ready=False,
                interpretation="Descriptive complete-price forecast only; freeze seed inventory, "
                               "configuration, caps and total allocation separately before execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prices = sub.add_parser("prices")
    prices.add_argument("--price", type=Path, action="append", required=True)
    prices.add_argument("--replications", type=int, required=True)
    prices.add_argument("--available-gpu-seconds", type=float, required=True)
    prices.add_argument("--output", type=Path, required=True)
    delivery = sub.add_parser("delivery")
    delivery.add_argument("--design", type=Path, required=True)
    delivery.add_argument("--outcomes", type=Path, required=True)
    delivery.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prices":
        result = price_report(args.price, replications_per_family=args.replications,
                              available_gpu_seconds=args.available_gpu_seconds)
    else:
        design = json.loads(args.design.read_text())
        result = delivery_report(**design, outcomes=json.loads(args.outcomes.read_text()))
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("delivery_criterion_passed", "missing_cases", "point_forecast_seconds",
                                 "point_forecast_fits_budget", "release_ready")}))


if __name__ == "__main__":
    main()
