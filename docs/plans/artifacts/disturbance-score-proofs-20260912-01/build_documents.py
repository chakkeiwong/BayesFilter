"""Build the integrated manuscript and a reading copy of its new derivation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
LIVE = ROOT / "docs/papers/ledh_younis_kdm_score"
NAME = "ledh_younis_kdm_score"
READING = "disturbance_score_proposal"
source = (LIVE / (NAME + ".tex")).read_text()
body = source.split(
    r"\subsection{A disturbance-coordinate proposal and its score}", 1
)[1].split(r"\subsection{Using both past and future information}", 1)[0]
preamble = source.split(r"\title{", 1)[0]
reading = preamble + r"""
\title{Likelihood Scores for Degenerate State-Space Models\\
Disturbance Proposals, Analytical Recursions, and Exact Control Variates}
\author{BayesFilter research note}
\date{12 September 2026}
\begin{document}
\maketitle
\begin{abstract}
When transition noise has lower dimension than the state, an ordinary
transition-density smoothing recursion can be undefined. We derive a score
construction in the disturbances that generate the state. Explicit
assumptions, propositions, and proofs establish the complete-path score,
unbiased particle likelihood and derivative estimates, an analytically
recursive implementation, and exactly centered reference control variates.
Conditional integration and a moving-support Gaussian example clarify the
mechanism. These results preserve the original model, but do not establish
an unbiased finite-particle normalized score or low variance at long horizons.
\end{abstract}
\section{The disturbance-coordinate proposal}
""" + body + r"""
\clearpage
\bibliographystyle{plainnat}
\bibliography{ledh_younis_kdm_score}
\end{document}
"""
(LIVE / (READING + ".tex")).write_text(reading)
build = ART / "final-build"
build.mkdir(exist_ok=True)
for name in [NAME, READING]:
    shutil.copy2(LIVE / (name + ".tex"), build / (name + ".tex"))
shutil.copy2(LIVE / (NAME + ".bib"), build / (NAME + ".bib"))
runs = []
for name in [NAME, READING]:
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
        ["bibtex", name],
    ] + [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"]
    ] * 3
    for i, command in enumerate(commands, 1):
        log = build / f"{name}-pass-{i}.stdout.log"
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=build, stdout=stream, stderr=subprocess.STDOUT
            )
        runs.append({
            "command": command, "cwd": str(build), "exit_code": result.returncode,
            "wall_seconds": time.monotonic() - start, "log": str(log),
        })
        if result.returncode:
            raise RuntimeError(f"Build failed; inspect {log}")
    shutil.copy2(build / (name + ".pdf"), LIVE / (name + ".pdf"))
    subprocess.run(
        ["pdftotext", "-layout", name + ".pdf", name + ".txt"],
        cwd=build, check=True
    )
(ART / "final-build-commands.json").write_text(json.dumps(runs, indent=2) + "\n")
manifest = {
    "integrated_tex_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "reading_copy_generated_from_section": "sec:disturbance-proposal",
    "derivation_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "files": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for name in [NAME, READING]
        for suffix in [".tex", ".pdf"]
        for path in [LIVE / (name + suffix)]
    },
}
(ART / "document-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({
    "status": "PASS", "builds": 2, "commands": len(runs),
    "wall_seconds": sum(r["wall_seconds"] for r in runs),
    "manifest": str(ART / "document-manifest.json"),
}))
