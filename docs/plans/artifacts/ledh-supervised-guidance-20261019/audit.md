# Mathematical and implementation audit

## Checked derivations

- Moment risk: expand the weighted square, then condition on C,U. The cross
  term vanishes by label/noise conditional independence. Explicit ordinary
  square integrability was added: finite weighted moments alone would not
  define the conditional mean when W is singular. W may depend on C; it may
  not use the latent label beyond C.
- Score risk: differentiation under the marginal-density integral yields
  Fisher's identity. Apply the previous result with the fixed-path joint score.
  The proof allows biased finite-particle estimates; it does not prove any
  optimized candidate improves a particular realized dataset.
- SIR labels: transition covariance I and parameter-independent initial law
  remove covariance/initial score terms; observation variance is
  100 exp(2 theta_3), giving residual-square/variance minus 9 per observation.
  The shared model observes interleaved infectious coordinates 1,3,...,17.
  No clipping or differentiation through the simulated trajectory is used.
- Gaussian product: covariance includes within- and between-component terms.
  The actual draw and importance denominator are the moment-matched Gaussian,
  not the original mixture. The total-precision positive-definiteness condition
  is necessary and sufficient on all of Euclidean space.

## Source and code boundaries

The original cSMC ADP loss and KDM NLL/optional mean loss remain source results.
Their score-supervised downstream selection and the SIR RBF adapter are local
extensions, not original-author replication claims. Technical paper sections
and local author source anchors were inspected before drafting.

Thirty-four distinct CPU diagnostic tests pass: exact risk decomposition, fixed-path SIR
label finite differences, data-partition rejection, mixture draw/density
tangents, existing default-guide tests, full shared-executor product-mixture
parity, nonlinear-flow and reset/correction call-chain tests. The first test
run caught a batched-label callback misuse and incorrect test keyword names;
both were repaired at their source. The final additions check Gaussian first
and second moments, admissible indefinite quadratic information, rejection of
nonpositive total precision, dataset-level averaging and missing/invalid pairs.
The 30-test regression, four-test final label suite (three repeated tests),
and three reporting tests are logged alongside this note.

No runtime NumPy numerical path or pfor was added. The pilot uses explicit
stable tf.function signatures and the existing shared analytical recursion.
Guide parameters are held fixed during score differentiation. Runtime checks
reject invalid factorizations instead of changing the guide silently.

## External audit limitations

MathDevMCP document audit failed internally at read_code despite the TeX file
existing. The direct probability statement was not encodable by its router.
A narrowed scalar squared-loss expansion and scalar completion of the square
were certified by the SymPy backend (mathdev-algebra.json and
mathdev-quadratic-completion.json). These certify the displayed algebra only,
not conditional expectation theory or the whole manuscript. mathdev-risk.json
preserves the unencodable probability result. The additional Gaussian score
derivative request was also unencodable by the tool's expression parser;
fixed-path finite differences and direct derivation check it independently.
Claude's bounded read-only review could not authenticate
(Not logged in). Neither failure is an approval or a mathematical refutation.
Direct derivation review and executable checks support the bounded pilot;
independent human mathematical review remains outstanding.

## Plan review

Rejected the prior three-density diagnostic as the consumer for a full-LEDH
claim. The plan instead invokes the shared UKF/flow/Contract-E/dual-cap
correction executor. Rejected the clipped author simulator for these labels.
Separated label-law validity, implementation validity and scientific benefit.
The finite bank is explicitly not neural training; short-horizon loss is not
T50 tuning; four test sequences cannot establish a ranking. Half-strength is
a bank member, so it cannot masquerade as an independent held-out adversary.
These repairs are recorded before the research pilot.

## Terminal math-to-code and result review

| Manuscript statement | Executable evidence | Verdict and limit |
|---|---|---|
| Moment-risk orthogonality | Discrete independent-label and exact Gaussian-quadrature tests | Correct under stated information/integrability assumptions |
| Fixed-path SIR labels | Shared transition tangent plus finite differences of joint log density | Correct for the specified continuous SIR scope; not clipped-model scores |
| Product covariance includes between-component dispersion | General mixture moment tests and draw/density tangent tests | Correct Gaussian product moments; sampled density is Gaussian |
| Indefinite guide precision can be admissible | Explicit negative-information draw, density tangent and rejection tests | Correct after optional flag repair; old default preserved |
| Full shared analytical consumer executes | Direct full-executor parity, flow/correction/reset wiring tests and six GPU finite differences | Checked for tested scopes, not all model/backend lanes |
| Loss selection uses separated data | Partition tests, saved labels, frozen calibration scales and terminal loss arithmetic | Checked; family nominations do not beat the unguided validation baseline |
| Score improvement | Four held-out datasets at each short horizon | Unsupported as a statistical ranking |

The old T2 raw rejection was wrong relative to the proposition's total-precision
condition. It has been repaired and is not represented as a negative result
for raw guidance. T5 has all 96 rows valid. terminal-audit.json verifies unique
keys, paired losses, source hashes and derivative artifacts. The largest
T5 directional discrepancy passes the declared 1e-4 relative-scaled tolerance;
the note does not imply roundoff-level agreement.

The abstract and the subsequent-comparisons transition now identify the new
supervised full-executor stage and distinguish it from older diagnostic
consumers. No old theorem, derivation or empirical finding was deleted. This
review covers the new propositions, their dependencies, implementation and
current-result claims; it is not a fresh independent proof certification of
every historical theorem in the 85-plus-page manuscript.

The result note separates engineering correctness, numerical validity and
scientific inference. Baseline loss is lowest on T5 validation. Favorable test
means are dominated by one dataset and the noise-score coordinate; every guide
loses on a different test dataset. No ranking, default, canonical admission or
HMC claim is issued. More independent observations and fresh scope-specific
calibration are the next discriminating evidence, not more particle replicates
treated as independent datasets.

## Rendered manuscript inspection

The final PDF has 86 pages (724678 bytes). Two pdflatex passes completed;
the saved pdf-build.log has no LaTeX warnings, unresolved references, or
overfull/underfull box reports. The title page and all six pages of the new
section (75--80) were rendered and visually inspected: equations, algorithm
steps, and the result table are legible and unclipped. The protected
81-page baseline is /tmp/ledh-younis-before-supervision-20261019.pdf. Existing
mathematical material remains in place; this inspection is not independent
certification of the older manuscript or a substitute for human feedback.
