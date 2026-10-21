"""Read-only, CPU-only audit of real preparation files; no numerical kernels."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_five_model_master as master

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
start = time.monotonic()
campaign = ROOT/"docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01"
roots = [campaign/"literature-prior-lgssm-advance-01",
         campaign/"literature-prior-other-four-advance-01"]
prepared = master.advance_prepared_selections(roots, master.MODELS, master.LITERATURE_PRIOR)
rows = []
for name, item in prepared.items():
    pilot = master.read(item["pilot"])
    selection = master.read(item["selection"])
    scope = item["scope"]
    rows.append(dict(model=name, selection=item["selection"], selection_hash=item["selection_hash"],
        prior=scope["prior"]["id"], particles=scope["particles"], horizon=scope["horizon"],
        dtype=scope["dtype"], pairwise_dtype=scope["pairwise_dtype"], tf32=scope["tf32"],
        controls=selection["selected"]["controls"], mechanics_status=pilot["status"],
        finite=pilot["finite"], divergences=pilot["divergences"],
        force_fallback_count=pilot["force_fallback_count"],
        acceptance_per_chain=pilot["acceptance_per_chain"],
        first_compilation_and_execution_seconds=pilot["first_compilation_and_execution_seconds"],
        warmed_two_step_seconds=pilot["transition_wall_seconds"],
        posterior_sampling_complete=False))
record = dict(status="PASS_REAL_FIVE_SCOPE_HANDOFF", checked_utc=master.stamp(),
    cpu_only=True, gpu_intentionally_hidden=True, command=[sys.executable, *sys.argv],
    wall_seconds=time.monotonic()-start, plan="docs/plans/surrogate-hmc-literature-priors-2026-10-15.md",
    rows=rows, nonclaims=["No posterior samples or convergence evidence",
        "Two-step timings include initialization; not long-chain forecasts",
        "Short mechanics acceptance is not a posterior-sampling acceptance estimate"])
master.write(args.output, record)
print(json.dumps(dict(status=record["status"], models=[r["model"] for r in rows],
                     output=str(args.output))))
