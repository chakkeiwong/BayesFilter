% Explicit CPU original-author-algorithm replication diagnostic.
cd('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev'); load_dir;
cd('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/octave_compat');
addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models'); addpath('/home/chakwong/BayesFilter-SQMC/third_party/audit/zhao_cui_tensor_ssm_p10/source/models/tensordot');
addpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/derived/models/sir_austria'); addpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/derived/models');
addpath(genpath('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/derived/octave_overrides'));
setenv('BAYESFILTER_REFERENCE_PROGRESS','/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/fit-progress.csv');
setenv('BAYESFILTER_INCREMENTAL_SMOOTHING','1');
setenv('BAYESFILTER_SMOOTH_TIMES','50');
setenv('BAYESFILTER_SMOOTH_SEED','3053');
setenv('BAYESFILTER_SMOOTH_N','100000');
setenv('BAYESFILTER_SMOOTH_REPETITIONS','1');
setenv('BAYESFILTER_REPEATED_TIMES','50');
fid=fopen('smoothing-summary.csv','w');
fprintf(fid,'time,ess,ess_fraction,max_weight,finite_fraction,seconds\n'); fclose(fid);
fid=fopen('smoothing-repetitions.csv','w');
fprintf(fid,'time,replicate,seed,ess,ess_fraction,max_weight,finite_fraction,seconds,zero_weight_count\n'); fclose(fid);
name='sir_austria'; d=0; m=18; n=9; T=50; N=5000;
rng(1); myModel=setup(ssmodel(name,d,m,n,T));
myModel.Y=dlmread('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/sir_austria-input-observations.csv')'; myModel.X=[];

previous=dlmread('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/sir_austria-probe-previous.csv')';
current=dlmread('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/sir_austria-probe-current.csv')';
if strcmp(name,'pp')
 predicted=predator_step(myModel,previous,repmat(myModel.pre.theta,1,size(previous,2)),'RK4');
else
 predicted=sir_step(previous,myModel.pre.theta);
end
pair=[current;previous];
actual=[predicted;reference_logprior(myModel,previous);reference_logtransition(myModel,pair,1);reference_loglike(myModel,pair,1)]';
expected=dlmread('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/sir_austria-probe-expected.csv');
errors=max(abs(actual-expected),[],1);
dlmwrite('target-parity-errors.csv',errors,'precision',17);
if any(~isfinite(errors)) || any(errors>1e-8), error('fixed target parity failed'); end

observations=myModel.Y; true_states=myModel.X; true_theta=myModel.theta;
if any(~isfinite(observations(:))) || any(~isfinite(true_states(:))), error('invalid data'); end
dlmwrite('observations.csv',observations','precision',17);
dlmwrite('true-states.csv',true_states','precision',17);
save('-mat7-binary','data.mat','observations','true_states','true_theta');
probe=priorsam(myModel,16); next=st_process(myModel,probe,1);
pair=[probe(1:d,:);next;probe(d+1:end,:)];
pdfs=[priorpdf(myModel,probe);transition(myModel,pair,1);like(myModel,pair,1)];
logs=[reference_logprior(myModel,probe);reference_logtransition(myModel,pair,1);reference_loglike(myModel,pair,1)];
mask=pdfs>realmin; log_error=max(abs(log(pdfs(mask))-logs(mask)));
if ~all(any(mask,2)) || ~isfinite(log_error) || log_error>1e-8, error('Gaussian log equivalence failed'); end
dlmwrite('gaussian-log-error.csv',log_error,'precision',17);
fid=fopen('call-chain.tsv','w');
for callback={'full_sol_reference','pre_sol_reference','TTSIRT','transition','priorpdf','sir_step','predator_step'}
 resolved=which(callback{1});
 if strcmp(resolved,'built-in function')
  if strcmp(callback{1},'TTSIRT'), resolved=file_in_loadpath('@TTSIRT/TTSIRT.m');
  else, resolved=file_in_loadpath([callback{1} '.m']); end
 end
 fprintf(fid,'%s\t%s\n',callback{1},resolved);
end
fclose(fid);
if ~strcmp(which('transition'),'/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-independent-validation-20261008-01/023-zhao-sir_d18-d26100831-fit53/derived/models/sir_austria/transition.m'), error('wrong transition'); end
if 0, fprintf('TARGET_PREPARATION_DONE\n'); return; end
poly=ApproxBases(Lagrangep(4,8),AlgebraicMapping(1),d+2*m);
opt=TTOption('tt_method','random','als_tol',1e-10,'local_tol',1e-4,'max_rank',20,'max_als',5,'init_rank',20,'kick_rank',5);
lowopt=TTOption('tt_method','random','als_tol',1e-10,'local_tol',1e-4,'max_rank',20,'max_als',5,'init_rank',20,'kick_rank',5);
rng(53);
sol=full_sol_reference(myModel,1,poly,opt,lowopt,N,4);
fit_timer=tic; sol=solve(sol); fit_seconds=toc(fit_timer);
fit_ess=sol.ESS_all; fit_times=sol.FTT_time;
save('-mat7-binary','fit-diagnostics.mat','fit_ess','fit_times','fit_seconds');
if strcmp(getenv('BAYESFILTER_INCREMENTAL_SMOOTHING'),'0')
 for terminal=[50]
  reference_save_smoothing(sol,terminal);
 end
end
fprintf('PUBLICATION_REPLICATION_DONE\n');
