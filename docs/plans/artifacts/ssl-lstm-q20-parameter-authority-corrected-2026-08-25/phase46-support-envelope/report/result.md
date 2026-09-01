# v2.8 Fixed-Law Theta Support Envelope

Status: `PASS_V2_8_SUPPORT_ENVELOPE_REPORT`
Branch: `n512_c_outside_two_bank_scalar_envelope` (descriptive only)

| Source | rows | roots | ESS | neg-mode | theta mean[0] | proposal residual |
|---|---:|---:|---:|---:|---:|---:|
| authority | 256 | 122 | 0.952283 | 0.530069 | 0.289568 | 0.000e+00 |
| bank_a | 256 | 103 | 0.801812 | 0.756588 | 3.550030 | 0.000e+00 |
| bank_b | 256 | 128 | 0.946687 | 0.517590 | 1.013180 | 0.000e+00 |
| bank_c | 256 | 125 | 0.975794 | 0.565503 | 0.877022 | 0.000e+00 |
| bank_n512_a | 512 | 248 | 0.927380 | 0.403469 | 1.446191 | 0.000e+00 |
| bank_n512_b | 512 | 233 | 0.968359 | 0.501739 | 0.587732 | 0.000e+00 |
| bank_n512_c | 512 | 241 | 0.878600 | 0.503522 | -0.896535 | 0.000e+00 |

The envelope and pairwise metrics are finite empirical diagnostics. They do not establish common support, IID whitening, posterior correctness, or mode discovery.
