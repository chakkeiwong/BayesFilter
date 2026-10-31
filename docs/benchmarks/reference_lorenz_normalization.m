% CPU-only diagnostic: unchanged author consumer, constant-restoration identity.
args=argv(); root=args{1}; output=args{2};
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst/distributions'));
author=fullfile(root,'.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz');
addpath(author);
% Execute the preserved source snapshot; the author tree supplies only data.
snapshot=fullfile(output,'sources');
addpath(snapshot);
assert(strcmp(which('cSMC_Resample'),fullfile(snapshot,'cSMC_Resample.m')));
assert(strcmp(which('RungeKutta4'),fullfile(snapshot,'RungeKutta4.m')));
assert(strcmp(which('Systematic_Resampling'),fullfile(snapshot,'Systematic_Resampling.m')));
load(fullfile(author,'SimulatedData.mat'));
d=8; H=size(y,1); T=H-1; p=d-2; N=128;
assert(H==101 && size(y,2)==6);
P=diag(ones(1,d-1),-1); P(1,d)=1; PT=P'; PP=P*P;
drift=@(x) ((PT*x'-PP*x').*(P*x')-x'+4.8801)';
Initial=struct('Mean',zeros(1,d),'Cov',.01*eye(d),'Precision',100*eye(d),...
  'MeanPrecision',zeros(1,d),'LogDetCov',d*log(.01),'QuadForm',0);
Transition=struct('G',.001*eye(d),'Ginv',1000*eye(d),'LogDetG',d*log(.001));
Transition.Mean=@(t,x) RungeKutta4(x,.1,drift);
Parameters=struct('Particles',N,'Dimension',d,'Time',T,'Iterations',0,...
  'Approximator','quadratic','Adaptive','no');
Projection=[eye(p),zeros(p,d-p)];
variances=[1e-4,1e-2]; seeds=[20260929,20260930];
archive=cell(2,2,3); checks={}; index=0; started=tic;
for ivar=1:2
  variance=variances(ivar); constant=-p/2*log(2*pi*variance);
  A=.5/variance*(Projection'*Projection);
  G=inv(Transition.Ginv+2*A);
  policy=struct('A',zeros(H,d,d),'b',-y*Projection/variance,...
    'c',.5/variance*sum(y.^2,2),'G',zeros(T,d,d),'LogDetG',ones(T,1)*log(det(G)));
  for t=1:H, policy.A(t,:,:)=A; end
  for t=1:T, policy.G(t,:,:)=G; end
  for iseed=1:2
    for arm=1:3
      obs=struct('y',y);
      if arm==1, add_constant=0; else, add_constant=constant; end
      obs.LogDensity=@(t,x) -.5/variance*sum(bsxfun(@minus,y(t,:),x(:,1:p)).^2,2)+add_constant;
      active_policy=policy;
      if arm==3, active_policy.c=policy.c-constant; end
      rand('state',seeds(iseed)); randn('state',seeds(iseed));
      result=cSMC_Resample(Parameters,Initial,Transition,obs,active_policy);
      archive{ivar,iseed,arm}=result;
      assert(all(isfinite(result{1,2}.LogNormConst)));
    end
    baseline=archive{ivar,iseed,1}{1,2};
    for arm=2:3
      candidate=archive{ivar,iseed,arm}{1,2};
      path_delta=abs(candidate.Trajectory-baseline.Trajectory);
      path_error=max(path_delta(:));
      path_scaled=max((path_delta./(1e-9+1e-9*abs(baseline.Trajectory)))(:));
      ess_delta=abs(candidate.ESS-baseline.ESS);
      ess_error=max(ess_delta);
      ess_scaled=max(ess_delta./(1e-9+1e-9*abs(baseline.ESS)));
      lhs=candidate.LogNormConst(end)-baseline.LogNormConst(end); rhs=H*constant;
      shift_error=abs(lhs-rhs); shift_scaled=shift_error/(1e-9+1e-9*abs(rhs));
      base_lw=baseline.LogIncrementalWeight;
      next_lw=candidate.LogIncrementalWeight;
      base_weights=exp(bsxfun(@minus,base_lw,max(base_lw,[],1)));
      next_weights=exp(bsxfun(@minus,next_lw,max(next_lw,[],1)));
      base_weights=bsxfun(@rdivide,base_weights,sum(base_weights,1));
      next_weights=bsxfun(@rdivide,next_weights,sum(next_weights,1));
      weight_delta=abs(base_weights-next_weights);
      weight_error=max(weight_delta(:));
      weight_scaled=max((weight_delta./(1e-9+1e-9*abs(base_weights)))(:));
      weights_valid=all(isfinite(next_weights(:))) && max(abs(sum(next_weights,1)-1))<1e-12;
      ancestor_equal=isequal(candidate.Ancestry,baseline.Ancestry);
      passed=path_scaled<=1 && ess_scaled<=1 && shift_scaled<=1 &&...
        weight_scaled<=1 && weights_valid && ancestor_equal;
      index=index+1;
      checks{index}=struct('observation_variance',variance,'seed',seeds(iseed),...
        'arm',arm,'passed',passed,'trajectory_max_abs',path_error,'ess_max_abs',ess_error,...
        'normalized_weight_max_abs',weight_error,'ancestry_exact',ancestor_equal,...
        'source_logZ',baseline.LogNormConst(end),'normalized_logZ',candidate.LogNormConst(end),...
        'expected_shift',rhs,'observed_shift',lhs,'shift_abs_error',shift_error,...
        'minimum_ess',min(candidate.ESS));
    end
  end
end
summary=struct('status','PASS','scope','LORENZ_AUTHOR_CONSTANT_RESTORATION_IDENTITY',...
 'replication','INCOMPLETE','ranking','NOT_TESTED','cpu_only',true,'gpu_intentionally_hidden',true,...
 'checks',{checks},'backend','Octave unchanged author reference','octave_version',version,...
 'particles',N,'observations',H,'dimension',d,'iterations',0,'seeds',seeds,...
 'algorithm_seconds',toc(started),'author_entry',which('cSMC_Resample'),...
 'source_convention','Gaussian observation normalizers omitted',...
 'normalized_convention','source logZ - H*p/2*log(2*pi*observation_variance)');
for k=1:length(checks), if ~checks{k}.passed, summary.status='FAIL'; end; end
save('-mat7-binary',fullfile(output,'export.mat'),'summary');
save('-binary',fullfile(output,'author-results.octave'),'archive','Parameters','y','variances','seeds');
