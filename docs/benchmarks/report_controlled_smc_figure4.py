"""CPU-only diagnostic rendering of the frozen Figure 4 uncertainty results.

This does not rerun, tune, rank or admit an algorithm. It displays statistics
already computed under controlled-smc-figure4-execution-20260929.md.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    source = args.input.resolve()
    summary = json.loads((source / "summary.json").read_text())
    curves = json.loads((source / "curves.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if (summary["status"] != "PASS" or summary["completed_rows"] != 4000
            or manifest["status"] != "COMPLETE"):
        raise ValueError("Require the completed frozen 4000-row experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    x = curves["variances"]
    statistic = "log10_relative_variance_logZ"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    review = {}
    for column, (method, title) in enumerate((("csmc", "Controlled SMC, N=128, I=3"),
                                             ("bpf", "Bootstrap PF, N=5529"))):
        ax = axes[0, column]
        for label, prefix, color, marker in (("TensorFlow replication", "local", "#1261a0", "o"),
                                              ("Saved author results", "author", "#b45309", "s")):
            values = curves["curves"][prefix + "_" + method][statistic]
            low, high = values["pointwise_95_interval"]
            ax.plot(x, values["estimate"], color=color, marker=marker, markersize=3,
                    linewidth=1.3, label=label)
            ax.fill_between(x, low, high, color=color, alpha=.16, linewidth=0)
        ax.set_title(title)
        ax.set_ylabel("log10[var(log Z) / mean(log Z)²]")
        ax.legend(frameon=False, fontsize=9)
        contrast = curves["contrasts"]["local_" + method + "_minus_author_" + method]
        low, high = contrast["pointwise_95_interval"]
        ax = axes[1, column]
        ax.axhline(0, color="0.3", linestyle="--", linewidth=.9)
        ax.plot(x, contrast["estimate"], color="#1261a0", marker="o", markersize=3)
        ax.fill_between(x, low, high, color="#1261a0", alpha=.16, linewidth=0)
        ax.set_ylabel("Replication minus author statistic")
        ax.set_xlabel("Latent process variance")
        review[method] = dict(
            positive_pointwise_intervals=[q for q, lo in zip(x, low) if lo > 0],
            negative_pointwise_intervals=[q for q, hi in zip(x, high) if hi < 0],
            zero_in_pointwise_interval=[q for q, lo, hi in zip(x, low, high) if lo <= 0 <= hi],
            interpretation="Descriptive interval pattern; no simultaneous test or equivalence verdict")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlim(min(x), max(x))
    fig.suptitle("Neuroscience Figure 4: 100 independent trials per method and variance\n"
                 "Shading: pointwise 95% bootstrap intervals, 3999 resamples", fontsize=12)
    fig.savefig(args.output / "figure4-source-comparison.png", dpi=160)
    fig.savefig(args.output / "figure4-source-comparison.pdf")
    plt.close(fig)
    review["inference_status"] = dict(
        hard_veto_screen="See original numerical summary; completed experiment required",
        statistically_supported_ranking="No whole-method or simultaneous ranking issued",
        descriptive_only="Pointwise differences and interval patterns displayed without equivalence testing",
        default_readiness="Not established; guided and temporal heuristics remain incomplete",
        next_evidence="Resolve any source discrepancy; complete the master comparator matrix")
    (args.output / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    paths = [source / name for name in ("summary.json", "curves.json", "manifest.json")]
    paths.append(Path(__file__).resolve())
    report_manifest = dict(
        role="POST_RUN_CPU_DIAGNOSTIC_RENDERING_ONLY", command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=sys.version, matplotlib=matplotlib.__version__, cpu_only=True,
        gpu_intentionally_hidden=True, algorithm_rerun=False,
        seeds="No new randomness; source bootstrap streams preserved in the input manifest and snapshot",
        plan="docs/plans/controlled-smc-figure4-execution-20260929.md",
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        result_file=str(args.output / "review.json"), wall_seconds=time.monotonic()-start)
    (args.output / "report-manifest.json").write_text(json.dumps(report_manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), role=report_manifest["role"],
                          wall_seconds=report_manifest["wall_seconds"])))


if __name__ == "__main__":
    main()
