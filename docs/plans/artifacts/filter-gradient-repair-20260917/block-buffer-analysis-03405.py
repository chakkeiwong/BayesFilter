"""Post-run compiler-buffer attribution; no executable benchmark kernel."""
import collections
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).parent
run = root / "run-03405"
receipt = json.loads((run / "block-buffer-attribution.json").read_text())
cost = json.loads((run / "block-public-memory.json").read_text())
reports = []
for name in receipt["buffer_assignments"]:
    source = (run / name).read_text()
    allocations = re.findall(r"^allocation (\d+): size (\d+), ([^\n]*)", source, re.M)
    groups = collections.defaultdict(list)
    temporary = None
    for number, size, flags in allocations:
        kind = ("temporary" if "preallocated-temp" in flags else
                "constant" if "constant" in flags else
                "parameter" if "parameter" in flags else
                "output" if "maybe-live-out" in flags else "other")
        groups[kind].append(int(size))
        if kind == "temporary":
            assert temporary is None
            temporary = int(number)
    summary = {k: {"allocations": len(v), "bytes": sum(v),
                  "hypothetical_separate_256_byte_rounding": sum((x + 255) // 256 * 256 for x in v)}
               for k, v in groups.items()}
    assert sum(row["bytes"] for row in summary.values()) == 185891
    start = source.index(f"allocation {temporary}:")
    end = source.find("\nallocation ", start + 1)
    if end == -1:
        end = source.index("Total bytes", start)
    values = re.findall(r"value: <(.+?)> \(size=(\d+),offset=(\d+)\): ([^\n]+)", source[start:end])
    intervals = sorted({(int(offset), int(offset) + int(size)) for _, size, offset, _ in values})
    merged = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    arrays = [(int(size), shape, int(offset), value) for value, size, offset, shape in values if not shape.startswith("(")]
    reports.append({"path": name, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "allocations": summary, "temporary_value_count": len(values),
                    "temporary_unique_offsets": len({int(o) for _, _, o, _ in values}),
                    "temporary_address_union_bytes": sum(right-left for left, right in merged),
                    "largest_array_values": sorted(arrays, reverse=True)[:5],
                    "largest_tuple_bytes": max(int(size) for _, size, _, shape in values if shape.startswith("("))})
assert reports[0]["allocations"] == reports[1]["allocations"]
result = {"schema": "filter_block_buffer_analysis.v1", "run": 3405,
          "timing_eligible": False, "compiler_reports": reports,
          "allocator_stages": {k: v["gpu"] for k, v in cost["stages"].items()},
          "cost_cohort": "block-public-costs-gpu-03398.json",
          "interpretation": [
              "The enclosing compiler program plans a 171360-byte reusable temporary arena (92.18% of total planned bytes).",
              "Complete public histories are returned as 407 live-out allocations including the top-level tuple; their logical payload totals 10411 bytes.",
              "Compiler allocations and allocator peaks are different measurements; no exact per-buffer reconciliation is claimed.",
              "Hypothetical 256-byte rounding is an explanatory calculation, not an assertion about individual runtime allocation lifetimes.",
              "The observed peak stays 283136 bytes from cold through warm and reaches 284160 with changed-input storage; current bytes stay 4352 then 5376.",
              "Retain the measured transient cost for these bounded signatures; native host residency and actual consumer capacity remain separate gates."]}
with (root / "block-buffer-analysis-03405.json").open("x") as output:
    json.dump(result, output, indent=2, allow_nan=False)
    output.write("\n")
print(json.dumps({"path": str(root / "block-buffer-analysis-03405.json"),
                  "allocations": reports[0]["allocations"],
                  "temporary_address_union_bytes": reports[0]["temporary_address_union_bytes"],
                  "largest_array_bytes": reports[0]["largest_array_values"][0][0]}))
