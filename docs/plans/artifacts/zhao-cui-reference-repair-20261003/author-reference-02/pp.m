% Derived audit runner; reduced settings; pinned source is immutable.
cd('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev');
load_dir;
cd('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/octave_compat');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models/tensordot');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp');
addpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/derived/models/pp');
addpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/derived/models');
addpath(genpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/derived/octave_overrides'));
name = 'pp'; d = 6; m = 2; n = 2;
T = 2; N = 64;
rng(1);
myModel = setup(ssmodel(name, d, m, n, T));
myModel = complete(myModel);
observations = myModel.Y;
truth = myModel.theta;
dlmwrite('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/pp-observations.csv', observations', 'precision', 17);
if any(~isfinite(observations(:))), error('nonfinite source data'); end
% Check the stable log evaluation wherever the inspected PDF is positive.
probe = priorsam(myModel,8);
next_probe = st_process(myModel,probe,1);
pair_probe = [probe(1:d,:); next_probe; probe(d+1:end,:)];
pdfs = [priorpdf(myModel,probe); transition(myModel,pair_probe,1); like(myModel,pair_probe,1)];
logs = [reference_logprior(myModel,probe); reference_logtransition(myModel,pair_probe,1); reference_loglike(myModel,pair_probe,1)];
mask = pdfs > realmin;
if ~all(any(mask,2)), error('no healthy Gaussian equivalence probes'); end
log_error = max(abs(log(pdfs(mask))-logs(mask)));
if ~isfinite(log_error) || log_error > 1e-8, error('stable Gaussian log mismatch'); end
dlmwrite('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/pp-gaussian-log-error.csv',log_error,'precision',17);

fid = fopen('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/pp-call-chain.tsv','w');
for callback = {'full_sol_reference','TTSIRT','transition','priorpdf','st_process','reference_logtransition'}
    resolved = which(callback{1});
    if strcmp(resolved,'built-in function')
        if strcmp(callback{1},'TTSIRT'), resolved = file_in_loadpath('@TTSIRT/TTSIRT.m');
        else, resolved = file_in_loadpath([callback{1} '.m']); end
    end
    fprintf(fid,'%s\t%s\n',callback{1},resolved);
end
fclose(fid);
if ~strcmp(which('transition'),'/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/derived/models/pp/transition.m'), error('wrong transition callback'); end
if ~strcmp(file_in_loadpath('full_sol_reference.m'),'/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/derived/models/full_sol_reference.m'), error('wrong solver class'); end
poly = ApproxBases(Lagrangep(4, 8), BoundedDomain([-1, 1]), d + 2*m);
opt = TTOption('tt_method', 'random', 'als_tol', 1E-10, ...
    'local_tol', 1E-4, 'max_rank', 4, 'max_als', 1, ...
    'init_rank', 2, 'kick_rank', 1);
rng(2);
sol = full_sol_reference(myModel, 1, poly, opt, opt, N, 5);
if ~isa(sol,'full_sol_reference'), error('wrong solver instance'); end
sol = solve(sol);
rng(3);
[thetas, sams, w, proposal_history, lml, stats] = smooth(sol, N, T);
raw_weights = stats.raw_log_weight;
dlmwrite('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/pp-raw-log-weights.csv', raw_weights(:), 'precision', 17);
save('-mat7-binary', '/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/pp.mat', ...
    'observations', 'truth', 'thetas', 'sams', 'w', 'proposal_history', 'lml', 'stats');
fid = fopen('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/zhao-cui-reference-repair-20261003/author-reference-02/pp-summary.tsv', 'w');
fprintf(fid, 'corrected_logmeanexp\t%.17g\n', lml);
fprintf(fid, 'legacy_mean_log_weight\t%.17g\n', stats.legacy_mean_log_weight);
fprintf(fid, 'finite_fraction\t%.17g\n', stats.finite_fraction);
fprintf(fid, 'importance_ess\t%.17g\n', stats.importance_ess);
fprintf(fid, 'corrected_status\t%d\n', stats.corrected_status);
fprintf(fid, 'tt_normalizer_accumulator\t%.17g\n', sol.logmarginal_likelihood);
fclose(fid);
if stats.corrected_status ~= 1, error('invalid corrected log weights'); end
fprintf('ZHAO_CUI_REFERENCE_DONE model=%s\n', name);
