# Local step-size adaptation and the centered funnel

October 28, 2026 literature clarification. The owner's recollection of another
successful adaptation is plausible. Two primary sources explicitly demonstrate
local step-size adaptive NUTS on the centered Neal funnel. Identifying which
one the owner remembers remains unresolved. This clarification adds no sampler
run; it reads the saved bridge result and the published and archived sources.

Ordinary warm-up adaptation in our pinned CmdStan 2.37 comparison learns a
single step size and a fixed Euclidean metric for retained sampling. The
benchmark used native Stan adaptation, with delta=.8 and a diagonal metric.
The two methods below instead incorporate state-dependent step-size selection
into retained Markov transitions, together with a correction that accounts
for the selection. They are algorithm changes, not an omitted setting in
the benchmark's standard warm-up.

## Closest literature matches

Bou-Rabee, Carpenter, Kleppe, and Marsden (2024), *Incorporating Local
Step-Size Adaptivity into the No-U-Turn Sampler using Gibbs Self Tuning*,
arXiv:2408.08259v1: section 4.3, equations16–18 and Algorithm7 define the
conditional step-size selection and Metropolis-within-Gibbs correction.
AppendixA addresses the measure-preserving involution and reversibility.
Section5.3/Figures3–4 report 250,000 draws: a fixed-step NUTS comparator at
h=.25 misses the deep neck, while adaptive NUTS produces negative-tail draws
without the displayed drop-off. This is evidence of improved tail exploration
in that experiment, not a comparison with an optimally tuned CmdStan service
or a demonstration of our four-chain convergence screen at 2000 draws. The
figure's trajectory example is 11-dimensional; do not silently treat every
figure as an exact repeat of our funnel10 fixture.

Bou-Rabee, Carpenter, Kleppe, and Liu (2026), *The Within-Orbit Adaptive
Leapfrog No-U-Turn Sampler*, JMLR27, 25-1452: WALNUTS selects microstep
refinements within an orbit (section3, equation9), accounts for local selection
probabilities, and proves reversibility for the specified constructions
(section3.4, Theorem22). Section4.3/Figures13–14 report improved exploration
of the far negative funnel tail, including independent runs of 200,000 draws
and a one-million-iteration diagnostic run. The comparator in Figure13 has
a fixed step size chosen to approximately match gradient cost, not a fully
tuned CmdStan baseline. Crucially, the authors explicitly report slow mixing
for both samplers. AppendixC separately discusses macro-step and error-threshold
warm-up tuning. Successful neck exploration must not be equated with rapid
convergence or universal efficiency superiority.

## Author-source inspection and limits

The GIST author source `metropolized_adaptive_nuts/step_size_adapt_NUTS_metropolized.py`
selects a step size from the current state, repeats selection at the proposed
state, and uses the two conditional probabilities for acceptance. The earlier
`step_size_adapt_NUTS_coarse_fine.py` is a different prototype and is not
evidence that the Metropolis correction can be omitted.

WALNUTS author `WALNUTSpy/adaptiveIntegrators.py`, function
`adaptLeapFrogR2P`, performs forward and backward refinement searches and
returns a log-weight difference. Equation (9) defines a conservative energy
envelope over the micro-trajectory, while Remark 13 explicitly documents the
optimized implementation's absolute endpoint-energy difference and says this
deterministic selection change preserves reversibility. The inspected author
function uses that endpoint criterion in both its forward and reverse searches;
this is a disclosed implementation choice, not a paper-source discrepancy.

The pinned C++ Walnutpie path used by the October 28 bridge is a different
configuration from the paper's WALNUTS-R2P figures. Its `macro_step` selects a
refinement deterministically and checks reverse recoverability; its transition
uses the supplied macro step without the paper's ±20% per-integration-step
jitter. This makes the saved run a related deterministic local-adaptation
variant, not a reproduction of Figures 13–14. The variant mismatch could
contribute to performance differences, but the saved run does not establish it
as the cause of poor mixing. These source checks describe historical code and
do not authorize a local implementation or execution.

Both downloaded author files are text-only archival references, not runtime
dependencies.

Paper PDFs, extracted text, author source, and hashes are stored in
`.localresources/funnel-local-adaptation-20261028/`. The exact URLs are recorded
in its `sources.json`. The inspected GIST version is the August2024 v1, and
the WALNUTS version is the March2026 published paper.

The short centered-funnel result in the native bridge ([saved report](../plans/bayesfilter-native-stan-walnutpie-result-2026-10-28.md))
does not test the paper's WALNUTS-R2P configuration or its 200,000- and
one-million-iteration experiments. Walnutpie visited `v < -3` in 14.75% of
retained draws, close to the exact 15.87%, but its 500 retained draws had
maximum R-hat 1.426 and minimum bulk ESS 8.6. This is poor mixing despite
descriptive tail visitation. It does not contradict the paper's narrower
tail-exploration claim, and the paper's result does not imply that this short
run should pass convergence diagnostics. Under the current repository policy,
NUTS/WALNUTS remain literature and historical-result context only; this note
does not propose a new implementation, sampling run, benchmark, or admission
route.
