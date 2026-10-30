# KDM/IWSG first GPU validation

This is an independent K1/K2 mechanics subphase of the authorized master. It
does not substitute for Appendix B training or bearings density comparisons.

Question: does a batched Gaussian-mixture TF/XLA endpoint reproduce normalized
source densities and the gradients of IWSG resampling? Primary criterion:
fixed source samples, density/weight/gradient agreement within 1e-7 absolute
plus relative; numerical mixture integral one; stopped sample gradients.
No stochastic ranking. Failures veto this primitive's correctness and trigger
localization, not rejection of IWSG. Missing/full consumer validation remains
a veto on a trained-filter replication claim.

Paper anchors: Younis–Sudderth section 3 equations (14)–(15), Appendix B
equations (20)–(22). Source commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18:
src/models/kde_particle_filter/kde_particle_filter.py lines 674–811, especially
738–749; packages/kernel-density-estimator-bandwdidth-prediction/packages/
kernel_density_estimation/kernel_density_estimator.py lines 232–267.
Execute the source resample_particles method with a minimal scalar Gaussian
fixture and the real source KernelDensityEstimator. Do not reconstruct an
algebraic reference while claiming the source endpoint was tested.

At fixed sampled z, equation (15) is exp(log m_phi(z)-stop(log m_phi(z))).
The numerator differentiates through component locations, weights and bandwidth;
the sample and denominator are fixed. Source KDE uses normalized (w+1e-8),
and source resampling adds 1e-8 after the exponential. Both additions are
explicit source-comparison settings. They are not mathematically identical to
the zero-epsilon paper formula and must not silently become general defaults.

Use two batches, three Gaussian components, scalar states and source-seeded
samples. Locations [-1,0,2] / [-2,.3,1.7], weights [.2,.5,.3] / [.6,.1,.3],
bandwidths .4/.8 are constructed diagnostic fixtures, not Appendix B values.
Check both an unnormalized expectation and a self-normalized expectation of
z^2+.3z; the latter exposes the normalization derivative. A fixed-sample
finite-difference of an all-one weight is NOT an IWSG gradient reference:
the denominator must be held fixed under parameter perturbation.

NumPy/PyTorch are confined to the external CPU reference; the eligible primitive
uses native batched TF operations with stable XLA signatures. No optimization
update, scalar-loop training, pfor, copied source implementation, or full filter
claim. Exact Appendix B numerical coefficients, perturbations and schedule are
still unrecovered. A mixture-Kalman implementation and actual learning remain
required; this primitive does not close those gaps.

Budget: one combined reference/GPU diagnostic launch capped at 120 GPU seconds,
charged conservatively for the whole subprocess and within the existing 7200
GPU-second tranche. Preserve reference samples, gradients, source hashes,
kernel snapshots, device/memory policy and structured results in the master's
versioned attempt directory. Seed 20260929. Script is registered as k1_iwsg.

Skeptical review PASS for this limited question: baselines are actual author
code and analytic normalization; fixture choices are explicitly non-empirical;
the two epsilon modifications, fixed-denominator derivative and consumer scope
are exposed. Passing cannot certify a complete trained KDM filter, statistical
superiority, paper learning curves or default-readiness.

Attempt09 failed before sampling because the reference process resolved the
system libstdc++ instead of its tf-gpu2 environment dependency (CXXABI_1.3.15).
The retry sets LD_LIBRARY_PATH for that CPU subprocess only; no package changes
or scientific-setting changes. Charge every retry to the unchanged tranche.
The failed attempt also carries the wrong author_commit metadata (controlled
SMC); its snapshotted KDM sources remain identifiable. Future manifests correct
that field. This failed infrastructure attempt is not numerical evidence.
