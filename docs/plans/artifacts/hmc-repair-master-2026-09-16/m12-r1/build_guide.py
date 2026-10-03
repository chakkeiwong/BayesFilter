"""Build and render the official book with M12's metric-probe amendment."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
DOCS = REPO/"docs"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    began = time.monotonic()
    output = ROOT/"guide-r1"
    stage = Path("/tmp/bayesfilter-hmc-m12-guide-20260920-r1")
    if not args.render_existing:
        output.mkdir(exist_ok=False)
        stage.mkdir(exist_ok=False)
        for name in ("main.tex", "preamble.tex", "references.bib"):
            shutil.copyfile(DOCS/name, stage/name)
        for directory in DOCS.iterdir():
            if directory.is_dir():
                (stage/directory.name).symlink_to(directory, target_is_directory=True)
    inputs = [DOCS/name for name in ("main.tex", "preamble.tex", "references.bib")]
    for name in ("chapters", "appendices", "generated"):
        inputs.extend(sorted((DOCS/name).glob("*.tex")))
    before = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    if not args.render_existing:
        with (output/"build.log").open("x") as log:
            process = subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT, timeout=600.)
    else:
        process = subprocess.CompletedProcess(command, 0)
    if before != {str(p.relative_to(REPO)): sha(p) for p in inputs}:
        raise RuntimeError("guide source changed during build")
    if process.returncode:
        raise RuntimeError("guide build failed")
    for name in ("main.pdf", "main.log", "main.blg", "main.fls"):
        shutil.copyfile(stage/name, output/name)
    subprocess.run(["pdftotext", "-layout", str(output/"main.pdf"), str(output/"guide.txt")], check=True)
    pages = (output/"guide.txt").read_text().split("\f")
    rendered = []
    for label, needle in (("sequential-probe", "metric_probe_num_results"),):
        for number, page in enumerate(pages, 1):
            if needle in "".join(page.split()):
                prefix = output/f"{label}-page-{number}"
                subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number), "-scale-to", "1800",
                                "-png", "-singlefile", str(output/"main.pdf"), str(prefix)], check=True)
                rendered.append({"role": label, "physical_page": number,
                                 "image": str(prefix.with_suffix(".png").relative_to(REPO))})
                break
        else:
            raise RuntimeError("new guide text missing: " + needle)
    log = (output/"main.log").read_text(errors="replace")
    record = {"command": command, "working_directory": str(stage), "returncode": process.returncode,
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-began,
        "gpu_used": False, "device": "cpu_reference", "numerical_framework_imported": False,
        "render_existing": args.render_existing,
        "prior_build_charge_seconds": 600. if args.render_existing else 0.,
        "prior_failure": "post-build text locator did not normalize line wrapping" if args.render_existing else None,
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "result_file": "docs/plans/bayesfilter-hmc-repair-m12-result-2026-09-20.md",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": before, "script_sha256": sha(Path(__file__)), "source_unchanged_during_build": True,
        "pdf_sha256": sha(output/"main.pdf"), "output_summary": re.findall(r"Output written on .*", log),
        "unresolved_citation_keys": sorted(set(re.findall(r"Citation `([^']+)'", log))),
        "rendered_pages": rendered}
    (output/"build-manifest.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
