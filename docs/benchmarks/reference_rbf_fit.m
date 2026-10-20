% Independent corrected author reference. CPU-only diagnostic, never a runtime.
args=argv();output=args{1};snapshot=fullfile(output,'sources');addpath(snapshot);
data=load(fullfile(snapshot,'SimulatedData_SigmaSq_1.mat'));
N=512;H=length(data.y);M=16;y=data.y;assert(H==101);
Initial=struct('Mean',0,'Var',5,'Precision',.2,'MeanPrecision',0);
Transition=struct('G',10,'Ginv',.1,'LogG',log(10));
Transition.Mean=@(t,x).5*x+25*x./(1+x.^2)+8*cos(1.2*t);
Observation=struct();Observation.LogDensity=@(t,x)-.5*(y(t)-x.^2/20).^2-.5*log(2*pi);
Parameters=struct('Particles',N,'Knots',M,'Time',H-1,'Iterations',1,...
 'KnotsType','particles','BandwidthFactor',.3,'AuditOutput',output);
% Cheap actual-endpoint regression before the accelerated full source run.
Small=Parameters;Small.Particles=64;Small.Time=5;
rand('state',20261003);randn('state',20261003);A=cSMC_Resample_fit_unvectorized(Small,Initial,Transition,Observation);
rand('state',20261003);randn('state',20261003);B=cSMC_Resample_fit_reference(Small,Initial,Transition,Observation);
for rr=1:2
 assert(isequal(A{rr,2}.Trajectory,B{rr,2}.Trajectory));
 assert(isequal(A{rr,2}.Ancestry,B{rr,2}.Ancestry));
 assert(isequal(A{rr,2}.LogNormConst,B{rr,2}.LogNormConst));
end
fprintf('vectorized_reference_bitwise_parity_passed\n');fflush(stdout);
seeds=[20261006,20261007];cases=cell(1,2);times=zeros(1,2);
for b=1:2
 rand('state',seeds(b));randn('state',seeds(b));tic;
 corrected=cSMC_Resample_fit_reference(Parameters,Initial,Transition,Observation);times(b)=toc;
 rounds=cell(1,2);
 for r=1:2
  pol=corrected{r,1};smc=corrected{r,2};normal=zeros(H,N);component_uniform=zeros(H,N);uniform=zeros(H,1);
  for t=1:H
   if t==1,m=zeros(N,1);q=5;else
    m=Transition.Mean(t-1,smc.Trajectory(smc.Ancestry(:,t-1),t-1));q=10;
   end
   beta=pol.Bandwidth(t);knots=pol.Knots(:,t)';weights=pol.Weights(:,t)';
   denom=1+2*beta*q;v=q/denom;mu=(m+2*beta*q*knots)/denom;
   lp=log(weights)-beta/denom*(m-knots).^2;
   probs=exp(lp-max(lp,[],2));probs=probs./sum(probs,2);
   component=smc.AuditComponents(:,t);idx=sub2ind([N,M],(1:N)',component);
   normal(t,:)=(smc.Trajectory(:,t)-mu(idx))'/sqrt(v);
   cumulative=cumsum(probs,2);previous=[zeros(N,1),cumulative(:,1:end-1)];
   lo=previous(idx);hi=cumulative(idx);assert(all(hi>lo));component_uniform(t,:)=((lo+hi)/2)';
   lw=smc.LogIncrementalWeight(:,t);w=exp(lw-max(lw));w=w/sum(w);
   cumulative=cumsum(w);cumulative(end)=1;previous=[0;cumulative(1:end-1)];
   ancestor=smc.Ancestry(:,t);lo=max([0;N*previous(ancestor)-(0:N-1)']);
   hi=min([1;N*cumulative(ancestor)-(0:N-1)']);assert(hi>lo);uniform(t)=(lo+hi)/2;
  end
  rounds{r}=struct('beta',pol.Bandwidth','weights',pol.Weights','knots',pol.Knots',...
   'constant',pol.MaxLogPolicyValue','x',smc.Trajectory',...
   'log_incremental_weight',smc.LogIncrementalWeight','logZ',smc.LogNormConst',...
   'ess',smc.ESS','ancestors',smc.Ancestry'-1,'normal',normal,...
   'component_uniform',component_uniform,'uniform',uniform);
 end
 fitted=corrected{2,1};
 cases{b}=struct('seed',seeds(b),'rounds',{rounds},'fit_target',fitted.FitTarget',...
  'full_coefficients',fitted.FullCoefficients');
 save('-mat7-binary',fullfile(output,sprintf('source_seed%d.mat',seeds(b))),'corrected');
end
fixture=struct('scope','CORRECTED_RBF_FIT_FULL_HORIZON_REFERENCE','particles',N,...
 'observations',H,'components',M,'observation_variance',1,'initial_variance',5,...
 'transition_variance',10,'bandwidth_factor',.3,'y',y,'cases',{cases});
summary=struct('scope','RBF_FIT_REFERENCE','replication','INCOMPLETE','seeds',seeds,...
 'source_seconds',times,'vectorized_reference_bitwise_parity',true,'cpu_only',true,'gpu_intentionally_hidden',true);
save('-mat7-binary',fullfile(output,'export.mat'),'fixture','summary');
