% Derived audit runner; reduced settings; pinned source is immutable.
cd('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev');
load_dir;
cd('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/octave_compat');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models/tensordot');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria');
addpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/derived/models/sir_austria');
addpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/derived/models');
addpath(genpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/derived/octave_overrides'));
name = 'sir_austria'; d = 0; m = 18; n = 9;
T = 2; N = 64;
rng(1);
myModel = setup(ssmodel(name, d, m, n, T));
myModel = complete(myModel);
observations = myModel.Y;
truth = myModel.theta;
dlmwrite('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/sir_austria-observations.csv', observations', 'precision', 17);
if any(~isfinite(observations(:))), error('nonfinite source data'); end
poly = ApproxBases(Lagrangep(4, 8), AlgebraicMapping(1), d + 2*m);
opt = TTOption('tt_method', 'random', 'als_tol', 1E-10, ...
    'local_tol', 1E-4, 'max_rank', 4, 'max_als', 1, ...
    'init_rank', 2, 'kick_rank', 1);
rng(2);
sol = full_sol_reference(myModel, 1, poly, opt, opt, N, 4);
sol = solve(sol);
rng(3);
[thetas, sams, w, proposal_history, lml, stats] = smooth(sol, N, T);
raw_weights = stats.raw_log_weight;
dlmwrite('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/sir_austria-raw-log-weights.csv', raw_weights(:), 'precision', 17);
save('-mat7-binary', '/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/sir_austria.mat', ...
    'observations', 'truth', 'thetas', 'sams', 'w', 'proposal_history', 'lml', 'stats');
fid = fopen('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-smoke-05/sir_austria-summary.tsv', 'w');
fprintf(fid, 'corrected_logmeanexp\t%.17g\n', lml);
fprintf(fid, 'legacy_mean_log_weight\t%.17g\n', stats.legacy_mean_log_weight);
fprintf(fid, 'finite_fraction\t%.17g\n', stats.finite_fraction);
fprintf(fid, 'importance_ess\t%.17g\n', stats.importance_ess);
fprintf(fid, 'corrected_status\t%d\n', stats.corrected_status);
fprintf(fid, 'tt_normalizer_accumulator\t%.17g\n', sol.logmarginal_likelihood);
fclose(fid);
if stats.corrected_status ~= 1, error('invalid corrected log weights'); end
fprintf('ZHAO_CUI_REFERENCE_DONE model=%s\n', name);
