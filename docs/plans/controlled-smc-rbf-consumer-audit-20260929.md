# Multimodal RBF source consumer audit and repair

The next question is whether the pinned author consumer samples the Gaussian
mixture in published equation11 and Algorithm1 step2(a). This precedes a long
NNLS reference run. The author-hosted supplement section9.1 explicitly makes
both mixture probabilities and component means functions of the same parent.
Local published-text lines342–366,453–472 and supplement lines3544–3588 were
inspected. The pinned ToyMultimodal/cSMC_Resample.m lines110–119 instead sample
component probabilities from row n and use a Gaussian mean from Ancestor(n).

For base transition N(m(x),q), RBF policy exp(c) sum_j w_j exp(-beta(z-k_j)^2),
let v=q/(1+2 beta q), mu_j(x)=(m(x)+2 beta q k_j)/(1+2 beta q).
Component masses are proportional to
w_j exp[-beta(m(x)-k_j)^2/(1+2 beta q)], and the log integral is
c - log(1+2 beta q)/2 + logsumexp(log w_j-beta(m(x)-k_j)^2/(1+2 beta q)).
Completing the square gives these formulas. Both probabilities and means must
use x=X[Ancestor(n)]. Selecting probabilities from X[n] gives a different
proposal unless the rows coincide; the paper's potential then lacks the
necessary proposal-density correction. This is a source-code hypothesis,
not evidence that the paper's mathematical algorithm is wrong.

## Evidence contract and defaults

The exact comparator is the paper's conditional Gaussian mixture, with the
pinned source preserved as a separate diagnostic. Execute the full source
endpoint on a tiny N64,H6,M16,I1 fixture using the first six observations of
the supplied variance1 series and seed20261003. These sizes are diagnostic
conveniences; they do not approximate Table2 evidence. The source bandwidth
factor .3 and normalized observation density are retained. A separate N512
terminal NNLS cost probe uses a frozen standard-normal cloud and target from
the final supplied observation; it measures one solve, not total runtime.

1. Run the unchanged source, then an observability-only copy with the same seed.
   Require bitwise identical trajectories and log normalizers. Preserve the
   exact instrumentation diff and source hashes. Record the total-variation
   difference between probabilities actually used and probabilities of the
   selected ancestor at every transition.
2. An exact two-parent/two-component fixture must expose a nonidentity-ancestry
   mismatch; identity ancestry and uniform policy must be no-fire controls.
3. In a separate diagnostic copy, change only the probability row to
   Ancestor(n). Require the measured conditional mismatch to be zero. This is
   an explicitly corrected paper-target reference, not unchanged-author parity.
4. Build a paper-derived TF/XLA RBF forward consumer that gathers the parent
   before computing both probabilities and means. Require Gaussian-product
   identities, invalid-policy rejection, and full fixed-input replay against
   the corrected independent reference at1e-8 absolute +1e-9 relative, exact
   ancestry, a single trace and verified GPU memory growth. The fitting port
   remains separate work; imported source policies are fixtures only.

The primary diagnostic criterion is exact conditional-kernel agreement.
Mismatch is a promotion veto for the unchanged source route and a repair
trigger, not a continuation veto for the mathematical method. Corrupt source
snapshots, changing baseline inputs, nonfinite accepted states, invalid policy
mass, missing replay inputs or exhausted phase/campaign budget stop the run.
NNLS time, sparse support, ESS and likelihood differences are explanatory only.
No parameter selection, performance ranking, Table2 replication, training,
general default or HMC claim follows. Heuristic performance comparisons are
deferred to the empirical phase; these are analytic correctness controls.

Skeptical audit PASS for diagnosis: the two parent-dependent quantities are
identified separately, the actual source endpoint is executed, instrumentation
has a no-change check, and the correction is labeled. A literal source replay
cannot certify the paper's kernel if the indexing mismatch is confirmed.
Do not launch a long unchanged-source replication while that veto remains.

## Commands, limits and artifacts

Use the existing dispatcher with c1_rbf_consumer_audit (CPU only, GPU hidden,
60 seconds per attempt,120 aggregate, at most2 attempts) and c2_rbf_forward
(trusted GPU/XLA,120 seconds per attempt,240 aggregate, at most2 attempts).
Both draw from the existing CPU300/GPU18000 allocation and inherited balance;
raise the attempt ceiling from34 to36 for these two bounded phases. No new
compute grant is introduced. NNLS probe has an internal20-second cap, and a
timeout there is a cost finding, not permission for an unbounded run.

Versioned outputs remain under artifacts/controlled-smc-kdm-replication-20260928.
The source audit emits fixture.json, summary.json, source/instrumentation
snapshots and a MAT export; GPU validation emits full replay diagnostics and
device provenance. Master/checkpoint/result must record any confirmed source
defect and keep literal-source reproduction separate from the corrected target.

Attempt33 stopped at Octave RNG initialization before any source filter call.
This installed Octave lacks rng; use the same explicit rand/randn state APIs
as the successful Lorenz reference. This is a reference-environment repair,
not a random-seed selection. The0.365037947 CPU seconds remain charged.

Attempt34 confirms the mismatch:98 affected particle transitions, maximum
component-probability total variation0.9999971113; corrected reference zero.
Instrumentation leaves the original trajectories and likelihoods bitwise
unchanged. Identity ancestry and uniform policy both have zero mismatch.
The exact two-parent example changes the conditional mean by5.33333333.
One512-column NNLS solve took0.00231s; this is not a full-runtime estimate.
The GPU replay now uses the two corrected source rounds as two native batch
rows, and fresh streams20261005. Skeptical implementation audit PASS: parent
gather precedes both mixture probabilities and means, analytic Gaussian
product and density-ratio checks are included, and fitting remains absent.
