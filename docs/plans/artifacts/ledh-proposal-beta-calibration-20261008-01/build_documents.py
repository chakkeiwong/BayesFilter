"""Local documentation build; no scientific kernel or framework is imported."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("document", choices=["monograph", "standalone"])
parser.add_argument("--passes", type=int, default=3)
parser.add_argument("--skip-bibtex", action="store_true")
args = parser.parse_args()
assert 1 <= args.passes <= 4
root = Path(__file__).resolve().parent
docs = root.parents[2]
assert (docs / "main.tex").exists()
source, stem = (
    ("main.tex", "main") if args.document == "monograph" else
    ("papers/ledh_covariance_proposal_20261008.tex", "ledh_covariance_proposal_20261008")
)
build = root / (args.document + "-build")
build.mkdir(exist_ok=True)
relative = build.relative_to(docs)
commands = []
started = time.monotonic()
offset = len(list(build.glob("pdflatex-*.stdout.log")))

def run(command, log_name):
    before = time.monotonic()
    with (build / log_name).open("w") as out:
        result = subprocess.run(command, cwd=docs, stdout=out,
                                stderr=subprocess.STDOUT, timeout=420, check=False)
    commands.append({"command": command, "cwd": str(docs),
                     "returncode": result.returncode,
                     "seconds": time.monotonic() - before,
                     "stdout_log": str(build / log_name)})
    if result.returncode:
        print(json.dumps(commands[-1]))
        print((build / log_name).read_text(errors="replace")[-4500:])
        (build / "failed-build.json").write_text(json.dumps(commands, indent=2))
        raise SystemExit(result.returncode)

for iteration in range(args.passes):
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", "-recorder", "-output-directory=" + str(relative), source],
        "pdflatex-" + str(offset + iteration + 1) + ".stdout.log")
    if iteration == 0 and not args.skip_bibtex:
        run(["bibtex", str(relative / stem)], "bibtex-" + str(offset + 1) + ".stdout.log")

log = (build / (stem + ".log")).read_text(errors="replace")
report = {"document": args.document, "commands": commands,
          "seconds": time.monotonic() - started,
          "frameworks_imported": False, "gpu_used": False,
          "warnings": re.findall(r"[^\n]*Warning:[^\n]*", log),
          "overfull_hboxes": log.count("Overfull \\hbox"),
          "overfull_vboxes": log.count("Overfull \\vbox"),
          "underfull_hboxes": log.count("Underfull \\hbox"),
          "output": re.findall(r"Output written[^\n]*", log)}
(build / ("build-" + str(offset + 1) + "-result.json")).write_text(
    json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "commands"}, indent=2))
