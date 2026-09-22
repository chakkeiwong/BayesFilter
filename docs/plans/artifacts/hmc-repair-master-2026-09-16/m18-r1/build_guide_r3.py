"""Build the single official book and render the changed discussion for review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
DOCS = REPO / "docs"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    output = ROOT / "guide-r3"
    stage = Path("/tmp/bayesfilter-hmc-m18-guide-20260921-r3")
    output.mkdir(exist_ok=False)
    stage.mkdir(exist_ok=False)
    shutil.copyfile(DOCS / "main.pdf", output / "prior-main.pdf")
    for name in ("main.tex", "preamble.tex", "references.bib"):
        shutil.copyfile(DOCS / name, stage / name)
    for directory in DOCS.iterdir():
        if directory.is_dir():
            (stage / directory.name).symlink_to(directory, target_is_directory=True)
    inputs = [DOCS / name for name in ("main.tex", "preamble.tex", "references.bib")]
    for name in ("chapters", "appendices", "generated"):
        inputs.extend(sorted((DOCS / name).glob("*.tex")))
    before = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    with (output / "build.log").open("x") as log:
        process = subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT, timeout=600.)
    if before != {str(p.relative_to(REPO)): sha(p) for p in inputs}:
        raise RuntimeError("guide source changed during build")
    if process.returncode:
        raise RuntimeError("guide build failed")
    for name in ("main.pdf", "main.log", "main.blg", "main.fls"):
        shutil.copyfile(stage / name, output / name)
    subprocess.run(["pdftotext", "-layout", str(output / "main.pdf"), str(output / "guide.txt")], check=True)
    pages = (output / "guide.txt").read_text().split("\f")
    rendered = []
    for label, needle in (("sequential-wrapper", "Aseparateoptionalwrapper"),
                          ("m15-evidence", "ThelaterM15experiment"),
                          ("m17-evidence", "ThelaterM17matrix"),
                          ("coverage-explanation", "Automaticallyinferredsuiterequirements"),
                          ("kinks-and-jumps", "Pakman"),
                          ("frozen-preprocessing", "Gorinova")):
        candidates = [(number, page) for number, page in enumerate(pages, 1)
                      if needle.lower() in "".join(page.split()).lower()]
        if label != "sequential-wrapper":
            candidates = [(number, page) for number, page in candidates
                          if number > 420 and "Bibliography" not in page]
        if not candidates:
            raise RuntimeError("guide text locator found no page: " + label)
        for number, page in candidates[:1]:
            prefix = output / f"{label}-page-{number}"
            subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number), "-scale-to", "1800",
                            "-png", "-singlefile", str(output / "main.pdf"), str(prefix)], check=True)
            rendered.append({"role": label, "physical_page": number, "image": str(prefix.with_suffix(".png"))})
    log = (output / "main.log").read_text(errors="replace")
    unresolved = sorted(set(re.findall(r"Citation `([^']+)'", log)))
    if any(key in unresolved for key in ("Gorinova2020", "Pakman2014", "Afshar2015", "gandy2021mcmctesting")):
        raise RuntimeError("repaired source citations remain unresolved")
    record = {"command": command, "working_directory": str(stage), "returncode": process.returncode,
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "gpu_used": False, "device": "cpu_reference", "numerical_framework_imported": False,
        "plan_file": "docs/plans/bayesfilter-hmc-repair-m18-design-2026-09-21.md",
        "result_file": "docs/plans/bayesfilter-hmc-repair-m18-result-2026-09-21.md",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "environment": sys.executable, "source_sha256": before, "script_sha256": sha(Path(__file__)),
        "source_unchanged_during_build": True, "pdf_sha256": sha(output / "main.pdf"),
        "protected_baseline_sha256": sha(output / "prior-main.pdf"),
        "output_summary": re.findall(r"Output written on .*", log),
        "unresolved_citation_keys": unresolved, "rendered_pages": rendered,
        "installed": False, "rendered_visual_review_pending": True}
    (output / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
