"""CPU-only post-run warmup trace diagnostic; no posterior estimates or admission."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root/"docs/plans/artifacts/surrogate-hmc-m13-sequential-recovery-03/m13-public-tuning/tuning"
    reference_path = root/"docs/plans/artifacts/surrogate-hmc-m13-assessment-reference-20260926-02/worker/reference/reference.json"
    reference = json.loads(reference_path.read_text())
    progress = json.loads((source/"fresh-progress.json").read_text())
    nomination = json.loads((source/"nomination/manifest.json").read_text())
    def read(where, record):
        path = where/record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("checksum mismatch")
        return tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64)
    center = read(source/"nomination", nomination["tensors"]["center"])
    factor = read(source/"nomination", nomination["tensors"]["factor"])
    chunks = [read(source, record["archive"]["tensors"]["states"]) for record in progress["chunks"]]
    latent = tf.concat(chunks, axis=0)
    raw = center + tf.linalg.matmul(latent, factor, transpose_b=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    names = ("log process-noise scale", "log observation-noise scale")
    for d, name in enumerate(names):
        ax = axes[d]
        mean, sd = reference["mean"][d], reference["sd"][d]
        ax.axhspan(mean-sd, mean+sd, color="gray", alpha=.14, label="reference mean ± SD")
        ax.axhline(mean, color="black", lw=.8)
        for chain in range(4):
            ax.plot(raw[:,chain,d].numpy(), lw=.7, alpha=.8, label=f"chain {chain+1}")
        ax.set_ylabel(name)
    axes[0].legend(ncol=5, fontsize=8, loc="upper right")
    events = [x for x in progress["events"] if x.get("check_due") and x.get("rhat")]
    # The controller archives the modern R-hat as the maximum of both variants.
    for d, name in enumerate(names):
        axes[2].plot([x["count"] for x in events],
            [x["rhat"]["maximum"][d] for x in events],
            marker="o", label=name)
    axes[2].axhline(1.05, color="black", ls="--", label="readiness threshold")
    axes[2].set_ylabel("modern R-hat (latest 1000)")
    axes[2].set_xlabel("discarded warmup transitions per chain")
    axes[2].legend(fontsize=8)
    fig.suptitle("M13 previous attempt: healthy but insufficiently mixed warmup")
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/"warmup-trace.png", dpi=170)
    fig.savefig(args.output/"warmup-trace.pdf")
    result = dict(role="post-run explanatory diagnostic only", cpu_only=True,
        gpu_intentionally_hidden=True, posterior_qualified=False,
        source=str(source), reference_sha256=reference["reference_sha256"],
        recent_raw_means=tf.reduce_mean(raw[-1000:],0).numpy().tolist(),
        recent_raw_sds=tf.math.reduce_std(raw[-1000:],0).numpy().tolist(),
        chain_endpoint_positions=raw[-1].numpy().tolist(),
        plots=["warmup-trace.png","warmup-trace.pdf"])
    (args.output/"report.json").write_text(json.dumps(result, indent=2)+"\n")

if __name__ == "__main__":
    main()
