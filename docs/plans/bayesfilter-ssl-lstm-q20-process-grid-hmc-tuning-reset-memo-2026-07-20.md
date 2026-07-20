# SSL-LSTM q=20 Process-Grid HMC Tuning Reset Memo

Date: 2026-07-20

Status: `CLOSED_RESOURCE_PROJECTION_STOP`

The q=20 plain-target process-grid HMC tuning lane stopped before candidate
tuning. Trusted one/two/four-process GPU/XLA rate canaries showed essentially
constant aggregate throughput on one RTX 4080 SUPER. The complete Round-0 grid
projects to 11.98 hours with no evidence extensions and 29.43 hours if every
allowed extension fires, both above the eight-hour cap. No tuning survivor or
conditional HMC test exists.

This is a resource stop, not a tuning failure, HMC rejection, geometry verdict,
or NeuTra result. See the result note:

`docs/plans/bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-result-2026-07-20.md`.

Do not launch against later shared-worktree source using the old receipts. The
two/four-worker receipts bind execution signature
`ca66d7a7c124ac0a6657b9c55ac1ca540d3d4d2c7c5f5a190d47f012425fa079`;
refresh the rate after any target/filter/HMC/source change. Preserve the
separate entry condition that q=20 NeuTra-HMC needs two admitted target-matched
q=20 transports.
