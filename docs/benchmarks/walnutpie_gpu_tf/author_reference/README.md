# Author reference source

These files are unchanged copies from bob-carpenter/walnuts at commit
895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a. The upstream MIT license is retained.
The SHA-256 inventory is in provenance.json and is checked by the loader.

This is independent NumPy reference code for a standalone experiment. It is
not a BayesFilter runtime backend. Use the bounded funnel_master driver in the
parent directory rather than launching mainFunnel.py: the upstream script
runs one million iterations, writes files in the current directory, and has
a machine-specific import path. Its funnel10 is 11-dimensional (one scale
coordinate plus ten conditional Gaussian coordinates).

The headless adapter omits an unused matplotlib import in memory. It evaluates
the same normalized analytic density with a tested direct expression instead
of repeated scipy distribution calls. The separate author-corrected arm also
adds the missing cumulative log-weight update at the second backward leaf.
Each run preserves that exact patch. No copied file is modified by the adapter.

mainFunnelTransient.py starts at v=-30 with maxC=30. This is a distinct, costly
experiment; the port currently validates exponents only through 19. A shorter
v=-6 mechanics test is not an exact reproduction of that experiment.
