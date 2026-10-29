% Independent reference harness: executes the unchanged external author code.
% Model configuration follows Heng et al. Appendix I.3, not a runtime port.
args = argv();
root = args{1}; output = args{2}; mode = args{3};
addpath(fullfile(root, '.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root, '.localresources/code/octave-statistics-1.5.0/inst/distributions'));
author = fullfile(root, '.localresources/code/heng-controlled-smc/StateSpaceModels/NeuroScience');
addpath(author);
% Native Octave seeding: this installation lacks the rng wrapper.
rand('state',20260929); randn('state',20260929);
load(fullfile(author, 'Data_NeuroScience.mat'));
if strcmp(mode, 'pilot'), H=128; N=256; else, H=length(y); N=1024; end
y=y(1:H); I=2;
Initial=struct('Mean',0,'Cov',1,'Precision',1,'MeanPrecision',0,'LogDetCov',0,'QuadForm',0);
Transition=struct('F',.99,'G',.11,'Ginv',1/.11,'LogDetG',log(.11));
Transition.Mean=@(t,x) .99*x;
Observation=struct('y',y,'Const',zeros(H,1));
for t=1:H, Observation.Const(t)=log(nchoosek(50,y(t))); end
% Keep the source's direct logistic arithmetic in this reference configuration.
expit=@(x) exp(x)./(1+exp(x));
Observation.LogDensity=@(t,x) Observation.Const(t)+y(t)*log(expit(x))+(50-y(t))*log(1-expit(x));
Parameters=struct('Particles',N,'Dimension',1,'Time',H-1,'Iterations',I,'Approximator','quadratic');
tic; result=cSMC_Resample(Parameters,Initial,Transition,Observation,'demo');
algorithm_seconds=toc;
summary=struct('backend','Octave independent author reference','octave_version',version,
  'mode',mode,'seed',20260929,'particles',N,'observations',H,'iterations',I,
  'algorithm_seconds',algorithm_seconds,'cpu_only',true,'gpu_intentionally_hidden',true,
  'author_entry',which('cSMC_Resample'),'mvnrnd',which('mvnrnd'),'x2fx',which('x2fx'));
summary.logZ=zeros(1,I+1); summary.minimum_precision=zeros(1,I+1);
summary.minimum_ess=zeros(1,I+1); summary.minimum_design_rcond=zeros(1,I+1);
fixture=struct('y',y','ntrials',50,'alpha',.99,'variance',.11,'rounds',{{}});
for j=1:(I+1)
  policy=result{j,1}; smc=result{j,2};
  coeff=[reshape(policy.A,H,1),policy.b,policy.c];
  precision=[1;ones(H-1,1)/.11]+2*coeff(:,1);
  assert(all(isfinite(coeff(:))) && all(precision>0));
  assert(all(isfinite(smc.LogNormConst)) && all(isfinite(smc.LogIncrementalWeight(:))));
  summary.logZ(j)=smc.LogNormConst(end); summary.minimum_precision(j)=min(precision);
  summary.minimum_ess(j)=min(smc.ESS); conditions=zeros(H,1);
  for t=1:H
    x=smc.Trajectory(:,1,t); D=[ones(N,1),x,x.^2]; conditions(t)=rcond(D'*D);
  end
  summary.minimum_design_rcond(j)=min(conditions);
  if strcmp(mode,'pilot')
    fixture.rounds{j}=struct('policy',coeff,'x',squeeze(smc.Trajectory)',
      'ancestors',smc.Ancestry'-1,'log_incremental_weight',smc.LogIncrementalWeight',
      'logZ',smc.LogNormConst,'ess',smc.ESS,
      'shift',squeeze(smc.ShiftTransitionMean)','quadform',smc.QuadForm');
  end
end
summary.status='PASS'; summary.scope='SOURCE_EXECUTION_ONLY';
summary.replication='INCOMPLETE'; summary.ranking='NOT_TESTED';
if strcmp(mode,'pilot')
  save('-mat7-binary',fullfile(output,'export.mat'),'summary','fixture');
else
  save('-mat7-binary',fullfile(output,'export.mat'),'summary');
  save('-binary',fullfile(output,'author-result.octave'),'result','Parameters','y');
end
