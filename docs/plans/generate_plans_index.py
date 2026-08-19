#!/usr/bin/env python3
"""Generate INDEX.md and INDEX-FULL.md for docs/plans.

Read-only over the corpus: parses title, Date:, Status:/Verdict:, and
supersession banners from each top-level *.md file, groups files into
lineages by stripping the date and trailing role tokens from the filename
stem, and writes two generated indexes:

- INDEX.md      one row per lineage: file count, date span, latest file,
                latest status, staleness flags.
- INDEX-FULL.md one row per file, grouped by lineage, newest first.

Regenerate after adding plan documents:

    python docs/plans/generate_plans_index.py

Deterministic; stdlib only; never modifies any other file.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

PLANS_DIR = Path(__file__).resolve().parent
INDEX = PLANS_DIR / "INDEX.md"
INDEX_FULL = PLANS_DIR / "INDEX-FULL.md"
SELF_OUTPUTS = {INDEX.name, INDEX_FULL.name, "CONVENTIONS.md"}

DATE_RE = re.compile(r"[-_](20)?26[-_](\d{2})[-_](\d{2})(?=$|[-_])")
DATE_LINE_RE = re.compile(r"^Date:\s*(\d{4})[-_](\d{2})[-_](\d{2})")
STATUS_LINE_RE = re.compile(r"^(?:Status|Verdict):\s*(.+?)\s*$")
SUPERSEDED_RE = re.compile(
    r"\*\*\s*(?:Partially\s+)?Superseded", re.IGNORECASE
)
CORRECTION_RE = re.compile(r"\*\*\s*Correction\s+\d{4}-\d{2}-\d{2}", re.IGNORECASE)

# Trailing role tokens stripped (repeatedly) from the dateless stem to form
# the lineage key. Longest-first so compound roles strip as a unit.
ROLE_TOKENS = [
    "plan-review-result", "result-reset-memo", "plan-review", "reset-memo",
    "reboot-reset-memo", "review-ledger", "review-request", "review-reply",
    "review-record", "master-program-result", "master-program", "master-plan",
    "execution-checkpoint", "execution-result", "execution-ledger",
    "interim-result", "terminal-result", "canary-result", "smoke-result",
    "subplan", "runbook", "checkpoint", "handoff", "dossier", "amendment",
    "addendum", "manifest", "contract", "strategy", "ledger", "memo",
    "result", "review", "reply", "request", "record", "policy", "spec",
    "audit", "note", "lock", "plan", "reboot", "closeout", "response",
    "verdict", "final", "extension", "reassessment", "recovery",
    "pre", "post", "seed",
]
# Round/attempt suffixes that follow a role token in some dialects.
ROUND_RE = re.compile(
    r"-(?:round|r|attempt|phase|p|stage|cycle)[-_]?\d+[a-z]?$"
)


def read_head(path: Path, n: int = 45) -> list[str]:
    lines: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for _ in range(n):
                line = fh.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    except OSError:
        pass
    return lines


def parse_file(path: Path) -> dict:
    head = read_head(path)
    title = next(
        (l[2:].strip() for l in head if l.startswith("# ")), path.stem
    )
    date = None
    status = None
    for line in head:
        if date is None:
            m = DATE_LINE_RE.match(line.strip())
            if m:
                date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if status is None:
            m = STATUS_LINE_RE.match(line.strip())
            if m:
                status = m.group(1).strip("`").strip()
    if date is None:
        m = DATE_RE.search(path.stem.replace("_", "-"))
        if m:
            date = f"2026-{m.group(2)}-{m.group(3)}"
    blob = "\n".join(head)
    return {
        "name": path.name,
        "title": title,
        "date": date or "????-??-??",
        "sort_date": date or "0000-00-00",
        "status": status.replace("`", "") if status else None,
        "superseded": bool(SUPERSEDED_RE.search(blob)),
        "corrected": bool(CORRECTION_RE.search(blob)),
    }


def lineage_key(name: str) -> str:
    stem = name[:-3] if name.endswith(".md") else name
    stem = stem.replace("_", "-")
    stem = DATE_RE.sub("", stem)
    changed = True
    while changed:
        changed = False
        m = ROUND_RE.search(stem)
        if m:
            stem = stem[: m.start()]
            changed = True
            continue
        for tok in ROLE_TOKENS:
            if stem.endswith("-" + tok):
                stem = stem[: -(len(tok) + 1)]
                changed = True
                break
    return stem or name


def family_key(lineage: str) -> str:
    stem = lineage
    for prefix in ("bayesfilter-", "batched-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    tokens = stem.split("-")
    return "-".join(tokens[:2]) if len(tokens) > 1 else stem


def flags(info: dict) -> str:
    out = []
    if info["superseded"]:
        out.append("SUPERSEDED")
    if info["corrected"]:
        out.append("corrected")
    return ", ".join(out)


def esc(text: str) -> str:
    return text.replace("|", "\\|")


def main() -> None:
    files = sorted(
        p for p in PLANS_DIR.glob("*.md") if p.name not in SELF_OUTPUTS
    )
    infos = [parse_file(p) for p in files]

    lineages: dict[str, list[dict]] = {}
    for info in infos:
        lineages.setdefault(lineage_key(info["name"]), []).append(info)
    for members in lineages.values():
        members.sort(key=lambda i: (i["sort_date"], i["name"]), reverse=True)

    ordered = sorted(
        lineages.items(),
        key=lambda kv: (kv[1][0]["sort_date"], kv[0]),
        reverse=True,
    )
    families: dict[str, list[tuple[str, list[dict]]]] = {}
    for key, members in ordered:
        families.setdefault(family_key(key), []).append((key, members))
    fam_ordered = sorted(
        families.items(),
        key=lambda kv: (kv[1][0][1][0]["sort_date"], kv[0]),
        reverse=True,
    )
    stamp = _dt.date.today().isoformat()
    n_files = len(infos)
    n_lineages = len(ordered)

    header = (
        "<!-- GENERATED FILE — do not edit by hand. -->\n"
        "<!-- Regenerate: python docs/plans/generate_plans_index.py -->\n\n"
    )

    with INDEX.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("# docs/plans Index (lineage summary)\n\n")
        fh.write(
            f"Generated {stamp}. {n_files} documents in {n_lineages} "
            f"lineages across {len(fam_ordered)} families, newest first. "
            "A lineage groups files sharing a filename stem after stripping "
            "dates and role suffixes; a family groups lineages by their "
            "first two name tokens. Grouping is heuristic — verify before "
            "relying on it. 'Latest' is by Date: line (filename date as "
            "fallback) and is NOT an authority ruling — check the file's "
            "own status and any supersession banner. Full per-file "
            "listing: [INDEX-FULL.md](INDEX-FULL.md). Conventions for new "
            "documents: [CONVENTIONS.md](CONVENTIONS.md).\n\n"
        )
        for fam, fam_lineages in fam_ordered:
            n_fam_files = sum(len(m) for _, m in fam_lineages)
            fh.write(
                f"## `{esc(fam)}` "
                f"({n_fam_files} files, {len(fam_lineages)} lineages)\n\n"
            )
            fh.write(
                "| Lineage | Files | Span | Latest file | Latest status | Flags |\n"
            )
            fh.write("|---|---:|---|---|---|---|\n")
            for key, members in fam_lineages:
                latest = members[0]
                span = (
                    latest["date"]
                    if len(members) == 1
                    else f"{members[-1]['date']} .. {latest['date']}"
                )
                status = esc(latest["status"] or "—")
                if len(status) > 70:
                    status = status[:67] + "..."
                fh.write(
                    f"| `{esc(key)}` | {len(members)} | {span} "
                    f"| [{esc(latest['name'])}]({latest['name']}) "
                    f"| {status} | {flags(latest) or '—'} |\n"
                )
            fh.write("\n")

    with INDEX_FULL.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("# docs/plans Index (full listing)\n\n")
        fh.write(
            f"Generated {stamp}. {n_files} documents grouped into "
            f"{n_lineages} lineages, newest lineage first; files within a "
            "lineage newest first. Summary view: [INDEX.md](INDEX.md).\n\n"
        )
        for key, members in ordered:
            fh.write(f"## `{esc(key)}`\n\n")
            fh.write("| Date | File | Status | Flags |\n")
            fh.write("|---|---|---|---|\n")
            for info in members:
                status = esc(info["status"] or "—")
                if len(status) > 70:
                    status = status[:67] + "..."
                fh.write(
                    f"| {info['date']} | [{esc(info['name'])}]({info['name']}) "
                    f"| {status} | {flags(info) or '—'} |\n"
                )
            fh.write("\n")

    print(f"Wrote {INDEX.name}: {n_lineages} lineages")
    print(f"Wrote {INDEX_FULL.name}: {n_files} files")


if __name__ == "__main__":
    main()
