# P6 SIR-SGQF R1B GPU Attempt 01 Repair Record

Date: 2026-07-16

Attempt root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-01`

Classification: `EVIDENCE_HARNESS_JSON_TUPLE_LIST_COMPARISON`.

The GPU replay reconstructed typed target signature
`0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`,
matching the CPU reference. GPU/CPU value and score gaps were approximately
`8.03e-15` and `3.75e-13` scale-normalized, statuses matched, all compiled
outputs were GPU-resident, and memory growth passed.

The final harness check compared the in-memory identity payload directly to
the JSON-loaded CPU payload. Semantically identical tuples in memory and lists
in JSON compare unequal in Python, so the harness correctly withheld the
identity file and admission event but returned `BLOCK_SIR_SGQF_R1B`.

The repair compares the JSON-normalized payloads. No target, identity schema,
source closure, dataset, prior, chart, filter, audit point, threshold, device,
or numerical result changes. A fresh GPU root must reproduce the identity and
all parity checks before admission.
