% Independent CPU diagnostic of the UNCHANGED author adaptive consumer.
args=argv(); root=args{1}; output=args{2};
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst/distributions'));
snapshot=fullfile(output,'sources'); addpath(snapshot); cd(snapshot);
assert(strcmp(which('cSMC_Resample'),fullfile(snapshot,'cSMC_Resample.m')));
assert(strcmp(which('RungeKutta4'),fullfile(snapshot,'RungeKutta4.m')));
assert(strcmp(which('Systematic_Resampling'),fullfile(snapshot,'Systematic_Resampling.m')));
driver=fileread(fullfile(snapshot,'cSMC_VaryF.m'));
prefix=strsplit(driver,'% Vary F'); assert(numel(prefix)==2);
eval(prefix{1});
assert(d==8 && T==100 && N==512 && I==4);
assert(strcmp(Parameters.Approximator,'quadratic') && strcmp(Parameters.Adaptive,'yes'));
forcing_values=[2.5,4.8801,8.5]; seeds=[20260929,20260930]; cases={}; all_valid=true;
for forcing_index=1:numel(forcing_values)
 F=forcing_values(forcing_index);
 Lorenz=@(x)((PT*x'-PP*x').*(P*x')-x'+F)';
 Transition.Mean=@(t,x)RungeKutta4(x,interval,Lorenz);
 for seed_index=1:numel(seeds)
  seed=seeds(seed_index); rand('state',seed);randn('state',seed);
  item=struct('forcing',F,'seed',seed,'status','FAIL','stages',{{}});
  timer=tic; lastwarn('');
  try
   result=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
   [warning_text,warning_id]=lastwarn();
   item.last_warning=warning_text;item.last_warning_id=warning_id;
   item.refinements=size(result,1)-1; item.status='PASS';
   filename=sprintf('author-F%.4f-seed%d.mat',F,seed);
   save('-mat7-binary',fullfile(output,filename),'result','Parameters','F','seed','y');
   item.archive=filename;
   for stage=1:size(result,1)
    policy=result{stage,1}; smc=result{stage,2};
    vals=[policy.A(:);policy.b(:);policy.c(:);policy.G(:);policy.LogDetG(:);...
      smc.Trajectory(:);smc.LogIncrementalWeight(:);smc.ESS(:);smc.LogNormConst(:)];
    valid=isreal(vals) && all(isfinite(vals));
    minimum_eigenvalue=Inf;minimum_relative_margin=Inf;
    if valid
     for t=1:(T+1)
      if t==1,base=Initial.Precision;else,base=Transition.Ginv;end
      precision=base+2*squeeze(policy.A(t,:,:));
      eigenvalues=eig((precision+precision')/2);
      minimum_eigenvalue=min(minimum_eigenvalue,min(eigenvalues));
      minimum_relative_margin=min(minimum_relative_margin,min(eigenvalues)/max(abs(eigenvalues)));
     end
     valid=valid && minimum_eigenvalue>0;
    end
    stage_record=struct('stage',stage-1,'valid',valid,...
      'minimum_precision_eigenvalue',minimum_eigenvalue,...
      'minimum_relative_precision_margin',minimum_relative_margin,...
      'minimum_ess',min(smc.ESS),'final_logZ',smc.LogNormConst(end));
    item.stages{end+1}=stage_record;
    if !valid,item.status='FAIL';end
   end
  catch err
   item.status='FAIL';item.error=err.message;item.error_id=err.identifier;
   item.error_stack=err.stack;
  end
  item.seconds=toc(timer);cases{end+1}=item;
  all_valid=all_valid && strcmp(item.status,'PASS');
  progress=struct('completed_cases',numel(cases),'cases',{cases});
  save('-mat7-binary',fullfile(output,'progress.mat'),'progress');
  fprintf('F=%.4f seed=%d status=%s seconds=%.3f\n',F,seed,item.status,item.seconds);
 end
end
if all_valid,verdict='PASS';else,verdict='FAIL';end
summary=struct('status',verdict,'scope','UNCHANGED_AUTHOR_ADAPTIVE_LORENZ_CPU_REFERENCE',...
 'replication','INCOMPLETE','cpu_only',true,'gpu_intentionally_hidden',true,...
 'octave_version',version(),'checks',{{struct('name','all_six_native_cases_valid','passed',all_valid)}},...
 'cases',{cases},'author_entry',which('cSMC_Resample'),'rk4_entry',which('RungeKutta4'),...
 'resampling_entry',which('Systematic_Resampling'),'mvnrnd_entry',which('mvnrnd'),...
 'x2fx_entry',which('x2fx'),'squareform_entry',which('squareform'),...
 'rng_coupling','Octave native streams; not matched to TF attempt76');
save('-mat7-binary',fullfile(output,'summary.mat'),'summary');
