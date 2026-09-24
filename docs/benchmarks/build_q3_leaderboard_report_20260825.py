"""Q3 leaderboard report builder: aggregates stamped row artifacts.

G-5 quarantine: refuses any row artifact without an `alg1_conformance`
stamp. Emits the owner-facing markdown report with hard-vetoes-first
ordering and the inference-status table required by the program's Q3
contract.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

ROW_ORDER = [
    "linear2d", "dlgssm", "predator_prey", "ksc_sv",
    "generalized_sv", "austria_sir",
]


def main() -> None:
    board_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "q3_board",
    )
    rows = {}
    for name in ROW_ORDER:
        path = os.path.join(board_dir, f"row_{name}.json")
        if not os.path.exists(path):
            rows[name] = None
            continue
        payload = json.load(open(path))
        if not payload.get("alg1_conformance"):
            raise SystemExit(
                f"G-5 QUARANTINE: row {name} artifact is unstamped"
            )
        rows[name] = payload

    lines = []
    lines.append("# Q3 Canonical LEDH Leaderboard (2026-08-25)\n")
    lines.append(
        "Program: Q3 of the completion program; plan: "
        "`bayesfilter-q3-leaderboard-execution-plan-2026-08-24.md`. "
        "All rows G-5 stamped; seeding uses the fidelity-#7-fixed "
        "independent replication streams. N=1008 particles per arm; "
        "16 value seeds / 8 score seeds unless a row's manifest says "
        "otherwise.\n"
    )

    # CONFIGURATION STATUS FIRST (2026-08-25 rule): before any number,
    # say what program each cell ran and its tuning status; refuse
    # unlabeled cells (same quarantine mechanism as G-5).
    lines.append("## Configuration status (READ BEFORE ANY NUMBER)\n")
    lines.append(
        "A cell whose program is not `production` or whose tuning is "
        "UNTUNED must not be debugged or interpreted as the production "
        "algorithm's performance. Per the per-scope tuning rule, "
        "UNTUNED cells carry no per-model claims.\n"
    )
    lines.append("| row | cell | program | tuning |")
    lines.append("|---|---|---|---|")
    for name, payload in rows.items():
        if payload is None:
            continue
        r = payload["result"]
        cells = [
            (k, v) for k, v in r.items()
            if isinstance(v, dict) and (
                k in ("canonical_ledh", "bootstrap_pf")
                or k.startswith("score_dir")
            )
        ]
        for key, cell in cells:
            program = cell.get("program")
            tuning = cell.get("tuning")
            if key == "bootstrap_pf":
                program = program or "comparator (bootstrap PF)"
                tuning = tuning or "N/A (no tunables beyond N)"
            if not program or not tuning:
                raise SystemExit(
                    f"UNLABELED CELL: {name}/{key} lacks program/tuning "
                    "fields — refusing to build the report"
                )
            lines.append(f"| {name} | {key} | {program} | {tuning} |")
    lines.append("")

    # hard vetoes next
    lines.append("## Hard-veto screen\n")
    veto_lines = []
    for name, payload in rows.items():
        if payload is None:
            veto_lines.append(f"- {name}: ROW ABSENT (not run)")
            continue
        r = payload["result"]
        problems = []
        for arm, cell in r.items():
            if not isinstance(cell, dict):
                continue
            if arm in ("canonical_ledh", "bootstrap_pf") or (
                arm.startswith("score_dir")
            ):
                if not (
                    cell.get("all_finite", True)
                    and cell.get("all_valid", True)
                ):
                    problems.append(arm)
        if problems:
            veto_lines.append(
                f"- {name}: VETO — nonfinite/invalid arms: {problems}"
            )
        else:
            veto_lines.append(f"- {name}: clean")
    lines.extend(veto_lines)
    lines.append("")

    lines.append("## Value cells\n")
    lines.append(
        "| row | data | exact ref | canonical mean (spread) | "
        "bootstrap mean (spread) | UKF-GF |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, payload in rows.items():
        if payload is None:
            lines.append(f"| {name} | ABSENT | — | — | — | — |")
            continue
        r = payload["result"]
        exact = r.get("exact_kalman")
        c = r.get("canonical_ledh", {})
        b = r.get("bootstrap_pf", {})
        u = r.get("ukf_gaussian_filter", {})
        exact_s = f"{exact:.3f}" if exact is not None else "—"
        c_s = f"{c.get('mean', float('nan')):.3f} ({c.get('seed_spread', float('nan')):.3f})"
        if exact is not None and "abs_error_of_mean_vs_exact" in c:
            c_s += f", |err| {c['abs_error_of_mean_vs_exact']:.3f}"
        b_s = f"{b.get('mean', float('nan')):.3f} ({b.get('seed_spread', float('nan')):.3f})"
        if exact is not None and "abs_error_of_mean_vs_exact" in b:
            b_s += f", |err| {b['abs_error_of_mean_vs_exact']:.3f}"
        u_s = f"{u.get('value', float('nan')):.3f}"
        if "abs_error_vs_exact" in u:
            u_s += f", |err| {u['abs_error_vs_exact']:.4f}"
        lines.append(
            f"| {name} | {r.get('data','')} | {exact_s} | {c_s} | "
            f"{b_s} | {u_s} |"
        )
    lines.append("")

    lines.append("## Score cells (canonical analytical)\n")
    lines.append(
        "| row | cell | dir | mode | mean (seed spread) | "
        "self-consistency rel err (seed 0, kind) | exact ref |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, payload in rows.items():
        if payload is None:
            continue
        r = payload["result"]
        for key in r:
            if key.startswith("score_dir"):
                s = r[key]
                exact_ref = s.get("exact_kalman_fd_score_dir0")
                exact_s = (
                    f"{exact_ref:.4f} (|err of mean| "
                    f"{s.get('abs_error_of_mean_vs_exact', float('nan')):.4f})"
                    if exact_ref is not None else "—"
                )
                mode = (
                    f"annealed k={s['annealed_stages']}"
                    if s.get("annealed_stages", 1) > 1 else "plain"
                )
                kind = s.get("self_consistency_kind", "central_fd_seed0")
                lines.append(
                    f"| {name} | {key} | {s['direction_index']} | "
                    f"{mode} | "
                    f"{s['mean']:.3f} ({s['seed_spread']:.3f}) | "
                    f"{s['self_consistency_rel_err_seed0']:.2e} "
                    f"({kind}) | {exact_s} |"
                )
    lines.append("")
    lines.append(
        "Score-cell reading guide: self-consistency columns compare the "
        "analytical score against a reference derivative OF THE SAME "
        "estimator (central FD for plain cells; the autodiff oracle for "
        "annealed cells, since FD is invalid across resampling-boundary "
        "crossings) — machine-precision values prove the derivation, "
        "not unbiasedness. The exact-reference column shows that the "
        "particle score MEAN deviates from the exact score at claim "
        "scale: on dlgssm T=50 both modes deviate by ~4 (~30x seed-SE) "
        "with OPPOSITE signs (plain -11.5, exact -15.7, annealed "
        "-20.1). Finite-N score bias is estimator-variant-dependent; "
        "the Austria Fisher result (bias shrinking under annealing) "
        "does NOT transfer as a general rule, and score cells must not "
        "be read as unbiased score estimates. This is the board's "
        "standing caution for Q5.\n"
    )

    lines.append("## Inference-status table\n")
    lines.append(
        "| question | status |\n|---|---|\n"
        "| hard-veto screen | see screen above; a vetoed row's cells "
        "are not interpreted |\n"
        "| statistically supported ranking | NONE claimed — no "
        "predeclared uncertainty analysis ranks arms; exact-reference "
        "absolute errors are the only absolute claims |\n"
        "| descriptive-only differences | all cross-arm value/score "
        "differences without an exact reference; seed spreads are "
        "descriptive |\n"
        "| default-readiness | not established by this board (Q5 "
        "scope); per-scope tuning rule stands |\n"
        "| next evidence needed | SGQF/zhao-cui comparator hookup "
        "(slice B); paired-seed uncertainty analysis before any "
        "ranking language |\n"
    )
    lines.append(
        "## Notes\n\n- UKF-GF is a Gaussian-approximation comparator; "
        "on KSC (mixture) and gen-SV (heteroskedastic) rows it is "
        "density-misspecified (recorded in the row artifacts).\n"
        "- The Austria row runs the Q2-calibrated annealed lane "
        "(k=4, c=8); per-scope calibration applies to that row only.\n"
        "- Absent rows are reported ABSENT, not silently dropped.\n"
    )

    stamps = {
        name: payload["alg1_conformance"]
        for name, payload in rows.items() if payload
    }
    lines.append("## Stamps\n")
    for name, stamp in stamps.items():
        lines.append(f"- {name}: `{stamp}`")
    lines.append("")

    out_path = os.path.join(
        _ROOT, "docs", "benchmarks",
        "q3-canonical-leaderboard-report-2026-08-25.md",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"[report] {out_path}")


if __name__ == "__main__":
    main()
