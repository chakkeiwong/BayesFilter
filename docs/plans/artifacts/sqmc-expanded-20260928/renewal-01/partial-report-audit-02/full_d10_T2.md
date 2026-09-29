# full_d10_T2: actual Kalman and particle scores

Program: FP64 GPU/XLA reference comparison, TF32 disabled. Each route calibrates flow substeps 2 versus 8 within this exact scope. Full-model repair scopes also calibrate terminal balancing; the other numerical protections remain inherited hypotheses. Tuning paths are in audit.json. This is not FP32/TF32 production evidence or a statistically supported ranking. Reused pilot observations are disclosed in the run manifest. Invalid rows are retained for diagnosis and excluded from error summaries.

N=1020, d=10, T=2. Q scores use lower-Cholesky coordinates, with log diagonal; they are not derivatives with respect to covariance entries.

## Data 197001, filter 198001

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

No evaluated final cell for this pair.

## Data 197001, filter 198002

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

No evaluated final cell for this pair.

## Data 197002, filter 198001

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

No evaluated final cell for this pair.

## Data 197002, filter 198002

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

No evaluated final cell for this pair.

