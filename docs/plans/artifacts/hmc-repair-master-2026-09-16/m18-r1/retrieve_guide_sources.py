"""Bounded public-source fallback after ResearchAssistant HTTP406 failures."""
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[5] / ".localresources/guide-source-audit-20260921"
RECORDS = []


def fetch(url, name):
    path = ROOT / name
    if path.exists():
        return path.read_bytes()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,
                headers={"User-Agent": "Mozilla/5.0 (academic citation verification)"}), timeout=25) as response:
            data = response.read(15000000)
        path.write_bytes(data)
        RECORDS.append({"url": url, "path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "seconds": time.monotonic()-started})
        return data
    except Exception as exc:
        RECORDS.append({"url": url, "exception": type(exc).__name__, "message": str(exc),
                        "seconds": time.monotonic()-started})
        return b""


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    fetch("https://arxiv.org/pdf/1211.2046", "pakman-paninski.pdf")
    html = fetch("https://proceedings.mlr.press/v119/gorinova20a.html", "gorinova-page.html")
    links = re.findall(rb'href="([^"]+\.pdf)"', html)
    for link in links[:2]:
        fetch(link.decode(), "gorinova-" + ("supplement.pdf" if b"supp" in link else "paper.pdf"))
    index = fetch("https://papers.nips.cc/paper_files/paper/2015", "nips-2015.html")
    for href, title in re.findall(rb'<a href="([^"]+)"[^>]*>([^<]+)</a>', index):
        if b"Reflection, Refraction" in title:
            url = "https://papers.nips.cc" + href.decode() if href.startswith(b"/") else href.decode()
            page = fetch(url, "afshar-page.html")
            for link in re.findall(rb'href="([^"]+\.pdf)"', page)[:2]:
                url = "https://papers.nips.cc" + link.decode() if link.startswith(b"/") else link.decode()
                fetch(url, "afshar-" + ("supplement.pdf" if "supplement" in url.lower() else "paper.pdf"))
    with (ROOT / "retrieval-fallback-r1.json").open("x") as out:
        json.dump(RECORDS, out, indent=2)
    print(json.dumps(RECORDS, indent=2))


if __name__ == "__main__":
    main()
