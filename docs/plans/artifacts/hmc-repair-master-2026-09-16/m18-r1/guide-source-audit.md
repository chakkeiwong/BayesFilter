# M18 guide-source corrections

The three unresolved bibliography keys are now supported by inspected technical
sources. The correction also removes an unsupported general claim that
reflection/refraction automatically repairs a continuous gradient kink.
This is a documentation correction, not a sampler implementation or default
change. The official book build and rendered inspection remain part of M18.

Local source directory: `.localresources/guide-source-audit-20260921/`.
ResearchAssistant fetch attempts returned HTTP 406; the official publication
PDFs were retrieved directly and parsed locally. Its available pdftotext
parser supplied text with low-confidence metadata, so titles/authors and
technical sections were checked manually. A relative-path parsing failure
for Pakman was repaired with the absolute PDF path before reading it.

| Source | Inspected technical material | Local verdict and guide change |
| --- | --- | --- |
| Pakman and Paninski, JCGS 2014 | Section 2.1, equations 2.6--2.35; Sections 2.3--2.4; Section 4, equations 4.6--4.22 | Exact quadratic dynamics and reflection handle truncation constraints. The special Bayesian Lasso extension crosses continuous kinks by updating region-specific dynamics while position/velocity remain continuous. It is not a generic hard-max posterior integrator. |
| Afshar and Domke, NIPS 2015 | Sections 2--4 and Algorithm 1; Section 5 Lemmas 1--2/Theorem 1; Section 6; supplement Sections 1--3 | Refraction changes normal momentum according to the potential jump. For a regular crossing with zero jump, the momentum is unchanged. The volume proof covers affine partition boundaries. Experiments with actual jumps cannot establish an arbitrary continuous-kink repair. |
| Gorinova, Moore and Hoffman, ICML 2020 | Sections 5.1--5.2, Algorithms 1--2, Sections 6--7; supplement A's Gaussian conditioning derivation, B.2--B.4 interceptors, C's experiment protocol | VIP learns interpolation during variational preprocessing, then fixes coordinates during HMC. The ELBO may miss modes. Replace ambiguous 'adaptively interpolating' with this actual sequence. |

The first attempted Pakman identifier, arXiv:1211.2046, was wrong: it names an
unrelated graphene paper. It was rejected on title inspection and was never
used to support the guide. The correct paper is arXiv:1208.4118v3, confirmed
by title/authors and the authors' published package documentation. The rejected
download remains historical retrieval evidence under `pakman-paninski.pdf`;
use only `pakman-correct-paper.pdf` and `pakman-correct-parsed-r2.json`.
`pakman-retrieval-correction.json` records the distinction and checksum.

## Original implementation checks

The CRAN mirror of the authors' `tmg` package is stored at
`.localresources/tmg-20260921`, commit
`bd996adcc584886cc66fc9cedd5d1426e54f63b1`. `R/tmg.R:23` transforms the
precision matrix/constraints to the canonical Gaussian frame and calls the
C++ sampler. `src/HmcSampler.cpp:44` samples velocities, locates the earliest
constraint hit, follows sine/cosine dynamics and reflects the normal velocity
at lines 91--109. `man/rtmg.Rd` explicitly limits samples to the initial region
when support is disconnected. This audit checks the constrained-Gaussian
route; it does not establish implementation of the paper's Lasso extension.
The package's numerical retry behavior and fixed burn-in are not imported as
BayesFilter defaults.

The Gorinova author repository is stored at
`.localresources/autoreparam-20260921`, commit
`4d6938c322ea975d7db71fd3f09950e5c905330e`.
`program_transformations.py:475` binds learned parameters and its scalar-normal
branch at line 558 constructs the affine interpolation.
`inference.py:140` materializes optimized variational/reparameterization values;
`inference.py:198` constructs HMC with those fixed coordinates while adapting
step size separately. `experiments.py:559` similarly materializes the learned
parameters before the HMC stage beginning at line 580. These anchors support
the preprocessing claim. No performance finding or algorithm default is
transferred from this code.

Afshar's official publication page and supplemental zip were inspected. The
zip contains only `supplementary.pdf` (additional trajectories, tuning/runtime
figures and rejection/reflection/refraction counts), with no Java source.
No author implementation was available in those inspected publication
resources. The guide therefore cites the paper's actual algorithm and theorem,
not a locally verified implementation of RHMC.

## Local reasoning and review

For a fixed deterministic finite position-only force, a kick
`(q,p) -> (q,p-h*g(q))` and a drift `(q,p) -> (q+h*M^-1*p,p)` are invertible
shears. Each preserves phase-space volume; symmetric kick-drift-kick composition
is reversed by momentum negation and changing integration direction. Together
with an exact endpoint Metropolis energy calculation, these are the relevant
proposal properties. A measure-zero nonsmooth set is not by itself a proof
of those properties, integration accuracy, irreducibility or useful mixing.
This derivation explains the corrected guide sentence; it does not endorse
arbitrary event truncations or implementations with stateful force conventions.

The strongest alternative to the correction would be a source-specific
continuous-kink integration rule. Pakman Section 4 supplies such a rule for
a piecewise-quadratic special case, which is now stated explicitly. Afshar's
zero-jump momentum formula supplies no corresponding general correction.
The checked distinction supports the prose repair without rejecting the
papers' actual results. All previous equations and substantive qualifications
in the surrounding chapter are preserved. Inspect the rendered replacement
and references at the terminal official-book build.
