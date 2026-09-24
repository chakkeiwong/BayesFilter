"""Post-run diagnostic reporting of frozen A06 results; no runtime selection.

Reads the existing report and preserves its matched-sequence restriction. These
bars are descriptive and cannot establish inferential rankings. Uses only the
standard library for data handling and Matplotlib for the exported figure.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/plans/artifacts/observation-tt-independent-filtering-20260915-01/report.json"
report = json.loads(SOURCE.read_text())
rows = {(row["dimension"], row["method"]): row for row in report["tables"]}
methods = [
    ("transition", "Transition"),
    ("stationary_prior", "Stationary prior"),
    ("sgqf_gaussian", "SGQF current marginal"),
    ("sgqf_joint", "SGQF joint conditional"),
    ("tt_predictive", "Predictive-chart TT"),
    ("tt_guided", "Observation-guided TT"),
    ("tt_pair_block", "Original pair TT"),
    ("tt_sgqf_safeguard", "SGQF-initialized + selected TT"),
]
assert all((dimension, method) in rows for dimension in (1, 4) for method, _ in methods)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.3), sharey=True)
colors = ["#a5abb0", "#c3c7cb", "#7baac0", "#287c9e", "#b2a4c8", "#9273b0", "#6c8d77", "#277447"]
for ax, dimension in zip(axes, (1, 4)):
    values = [rows[dimension, method]["regimes"]["all"]["mean_mse"] for method, _ in methods]
    ids = report["matched_sequences"][str(dimension)]
    assert all(rows[dimension, method]["sequence_ids"] == ids for method, _ in methods)
    bars = ax.barh(range(len(methods)), values, color=colors, height=.66)
    ax.set_yticks(range(len(methods)), [label for _, label in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_xlabel("Normalized filtering mean-square error (lower is better)")
    ax.set_title(f"{dimension}D · {len(ids)} matched sequences", fontweight="bold", pad=12)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * .025, bar.get_y() + bar.get_height()/2,
                f"{value:.5f}", va="center", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle("A06 completed filtering comparison", fontsize=17, fontweight="bold", y=.97)
fig.text(.02, .055,
         "Descriptive means; no overall ranking is established. 12 sequences × 20 observations per dimension; 4 particle replicates, N=512.\n"
         "4D shows the same 11 successful sequences for every method. SGQF fails on sequence 8 of the full 12, blocking promotion.\n"
         "The selected TT also triggers two 1D conditional heuristic vetoes. See the result note for regime-specific uncertainty.",
         fontsize=9, linespacing=1.5)
fig.subplots_adjust(left=.245, right=.985, top=.84, bottom=.23, wspace=.24)
for extension in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"completed-filtering-results.{extension}", dpi=170, facecolor="white")
provenance = {
    "classification": "post-run diagnostic reporting only; no experiment or candidate selection",
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "metric": "all-observation normalized filtering MSE; equal sequence weighting",
    "matched_sequences": report["matched_sequences"],
    "plot_has_inferential_intervals": False,
    "candidate_advances": report["inference"]["candidate_advances"],
    "files": {f"completed-filtering-results.{ext}": hashlib.sha256((OUT / f"completed-filtering-results.{ext}").read_bytes()).hexdigest() for ext in ("png", "pdf", "svg")},
}
(OUT / "plot-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"status": "COMPLETE", "output_directory": str(OUT.relative_to(ROOT)), "matched_sequences": {key: len(ids) for key, ids in report["matched_sequences"].items()}}))
