"""Build and archive the guide in a clean stage, without stale source aux files."""
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
DOCS = REPO / "docs"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="r1", choices=("r1", "r2", "r3"))
    args = parser.parse_args()
    started = time.monotonic()
    stage = Path("/tmp") / ("bayesfilter-hmc-configuration-guide-20260918-" + args.revision)
    output = ROOT / ("guide-" + args.revision)
    stage.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
    for name in ("main.tex", "preamble.tex", "references.bib"):
        shutil.copyfile(DOCS / name, stage / name)
    for directory in DOCS.iterdir():
        if directory.is_dir():
            (stage / directory.name).symlink_to(directory, target_is_directory=True)
    inputs = [DOCS / name for name in ("main.tex", "preamble.tex", "references.bib")]
    inputs += sorted((DOCS / "chapters").glob("*.tex"))
    inputs += sorted((DOCS / "appendices").glob("*.tex"))
    inputs += sorted((DOCS / "generated").glob("*.tex"))
    before = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    with (output / "build.log").open("w") as log:
        process = subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT)
    after = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    if before != after:
        raise ValueError("Guide source changed during the build")
    if process.returncode:
        raise RuntimeError(f"Guide build failed: {process.returncode}; see {output / 'build.log'}")
    for name in ("main.pdf", "main.log", "main.blg", "main.fls"):
        shutil.copyfile(stage / name, output / name)
    subprocess.run(["pdftotext", "-layout", str(output / "main.pdf"),
                    str(output / "guide.txt")], check=True)
    pages = (output / "guide.txt").read_text().split("\f")
    needles = {"configuration": "hmc_configuration.py"}
    rendered = []
    # Exact page numbers are also recorded below; inspect these images manually.
    for label, needle in needles.items():
        for number, page in enumerate(pages, 1):
            if needle in page:
                prefix = output / f"{label}-page-{number}"
                subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number),
                                "-scale-to", "1800", "-png", "-singlefile",
                                str(output / "main.pdf"), str(prefix)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rendered.append({"role": label, "physical_page": number,
                                 "image": str(prefix.with_suffix(".png").relative_to(REPO))})
                if label != "coverage":
                    break
    log = (output / "main.log").read_text(errors="replace")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "working_directory": str(stage), "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - started, "gpu_used": False,
        "numerical_framework_imported": False, "seeds": "N/A: document build",
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "result_file": "docs/plans/bayesfilter-hmc-repair-m7-result-2026-09-17.md",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     check=True, capture_output=True, text=True).stdout.strip(),
        "tex_version": subprocess.run(["pdflatex", "--version"],
                                      check=True, capture_output=True, text=True).stdout.splitlines()[0],
        "source_sha256": before, "script_sha256": sha(Path(__file__)),
        "source_unchanged_during_build": True,
        "pdf_sha256": sha(output / "main.pdf"), "build_log_sha256": sha(output / "build.log"),
        "output_summary": re.findall(r"Output written on .*", log),
        "unresolved_citation_keys": sorted(set(re.findall(r"Citation `([^']+)'", log))),
        "rendered_pages": rendered,
        "render_review": "Manual visual review recorded in master result after this build",
    }
    with (output / "build-manifest.json").open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("elapsed_seconds", "output_summary",
          "unresolved_citation_keys", "rendered_pages")}, indent=2))


if __name__ == "__main__":
    main()
