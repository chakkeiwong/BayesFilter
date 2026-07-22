"""Build the exact Phase 9 nonlinear GPU/XLA command expansion."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = "docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla"
LOG_ROOT = "docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla"
PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu/bin/python"
RUNNER = "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py"
SEEDS = (81120, 81121, 81122, 81123, 81124)

ROWS = {
    "fixed-sir": {"full_time_steps": 20, "prefixes": (1, 5, 20), "gate_b_particles": 4},
    "predator-prey": {"full_time_steps": 20, "prefixes": (1, 5, 20), "gate_b_particles": 2},
    "actual-sv": {"full_time_steps": 1000, "prefixes": (4, 50, 250, 1000), "gate_b_particles": 4},
    "generalized-sv": {"full_time_steps": 1008, "prefixes": (4, 50, 252, 1008), "gate_b_particles": 4},
    "ksc-sv": {"full_time_steps": 1000, "prefixes": (4, 50, 250, 1000), "gate_b_particles": 4},
}


def _runtime_paths(
    gate: str,
    row: str,
    time_steps: int,
    num_particles: int,
    seed: int,
    stage: str,
) -> tuple[str, str, str]:
    directory = f"{ARTIFACT_ROOT}/{gate}/{row}" if gate == "gate-d" else f"{ARTIFACT_ROOT}/{gate}"
    stem = f"{row}-t{time_steps}-n{num_particles}-seed{seed}-{stage}"
    return f"{directory}/{stem}.json", f"{directory}/{stem}.md", f"{LOG_ROOT}/{gate}-{stem}.log"


def _runtime_command(
    gate: str,
    row: str,
    time_steps: int,
    num_particles: int,
    seed: int,
    stage: str,
) -> dict[str, Any]:
    output, markdown, log = _runtime_paths(gate, row, time_steps, num_particles, seed, stage)
    argv = [
        PYTHON,
        RUNNER,
        "--row",
        row,
        "--stage",
        f"{stage}-only",
        "--batch-seeds",
        str(seed),
        "--time-steps",
        str(time_steps),
        "--num-particles",
        str(num_particles),
        "--device-scope",
        "visible",
        "--cuda-visible-devices",
        "0",
        "--device",
        "/GPU:0",
        "--expect-device-kind",
        "gpu",
    ]
    score_reference = None
    if stage == "fd":
        score_reference = _runtime_paths(
            gate,
            row,
            time_steps,
            num_particles,
            seed,
            "score",
        )[0]
        argv.extend(("--score-reference-json", score_reference))
    argv.extend(("--output", output, "--markdown-output", markdown))
    return {
        "gate": gate,
        "row": row,
        "stage": f"{stage}-only",
        "seed": seed,
        "time_steps": time_steps,
        "num_particles": num_particles,
        "score_reference_json": score_reference,
        "output": output,
        "markdown_output": markdown,
        "log": log,
        "argv": argv,
        "shell_command": f"MPLCONFIGDIR=/tmp {shlex.join(argv)} > {shlex.quote(log)} 2>&1",
    }


def _aggregate_command(row: str) -> dict[str, Any]:
    full_time_steps = int(ROWS[row]["full_time_steps"])
    score_paths = [
        _runtime_paths("gate-d", row, full_time_steps, 10000, seed, "score")[0]
        if seed != SEEDS[0]
        else _runtime_paths("gate-c", row, full_time_steps, 10000, seed, "score")[0]
        for seed in SEEDS
    ]
    fd_paths = [
        _runtime_paths("gate-d", row, full_time_steps, 10000, seed, "fd")[0]
        if seed != SEEDS[0]
        else _runtime_paths("gate-c", row, full_time_steps, 10000, seed, "fd")[0]
        for seed in SEEDS
    ]
    directory = f"{ARTIFACT_ROOT}/gate-d/{row}"
    output = f"{directory}/{row}-full-five-seed-aggregate.json"
    markdown = f"{directory}/{row}-full-five-seed-aggregate.md"
    log = f"{LOG_ROOT}/gate-d-{row}-full-five-seed-aggregate.log"
    argv = [
        PYTHON,
        RUNNER,
        "--row",
        row,
        "--stage",
        "aggregate",
        "--batch-seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--time-steps",
        str(full_time_steps),
        "--num-particles",
        "10000",
        "--device-scope",
        "cpu",
        "--expect-device-kind",
        "cpu",
        "--score-shards",
        ",".join(score_paths),
        "--fd-shards",
        ",".join(fd_paths),
        "--output",
        output,
        "--markdown-output",
        markdown,
    ]
    return {
        "gate": "gate-d",
        "row": row,
        "stage": "aggregate",
        "seeds": list(SEEDS),
        "time_steps": full_time_steps,
        "num_particles": 10000,
        "score_shards": score_paths,
        "fd_shards": fd_paths,
        "output": output,
        "markdown_output": markdown,
        "log": log,
        "argv": argv,
        "shell_command": (
            f"CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp {shlex.join(argv)} "
            f"> {shlex.quote(log)} 2>&1"
        ),
    }


def build_manifest() -> dict[str, Any]:
    gate_b = []
    gate_c = []
    gate_d = []
    aggregates = []
    for row, row_spec in ROWS.items():
        gate_b_particles = int(row_spec["gate_b_particles"])
        gate_b.append(_runtime_command("gate-b", row, 1, gate_b_particles, SEEDS[0], "score"))
        gate_b.append(_runtime_command("gate-b", row, 1, gate_b_particles, SEEDS[0], "fd"))
        for time_steps in row_spec["prefixes"]:
            gate_c.append(_runtime_command("gate-c", row, int(time_steps), 10000, SEEDS[0], "score"))
            gate_c.append(_runtime_command("gate-c", row, int(time_steps), 10000, SEEDS[0], "fd"))
        full_time_steps = int(row_spec["full_time_steps"])
        for seed in SEEDS[1:]:
            gate_d.append(_runtime_command("gate-d", row, full_time_steps, 10000, seed, "score"))
            gate_d.append(_runtime_command("gate-d", row, full_time_steps, 10000, seed, "fd"))
        aggregates.append(_aggregate_command(row))
    return {
        "schema_version": "bayesfilter.ledh.phase9.exact_commands.v1",
        "working_directory": str(ROOT),
        "python": PYTHON,
        "runner": RUNNER,
        "artifact_root": ARTIFACT_ROOT,
        "log_root": LOG_ROOT,
        "seeds": list(SEEDS),
        "rows": {
            row: {
                **row_spec,
                "prefixes": list(row_spec["prefixes"]),
            }
            for row, row_spec in ROWS.items()
        },
        "gate_b_commands": gate_b,
        "gate_c_commands": gate_c,
        "gate_d_commands": gate_d,
        "aggregate_commands": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    payload = build_manifest()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if output.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"exact command manifest is stale: {output}")
        print(f"exact command manifest is current: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
