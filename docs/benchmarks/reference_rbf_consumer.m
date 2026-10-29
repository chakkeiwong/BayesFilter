% Independent CPU-only author-source diagnostic; no runtime implementation.
args=argv();root=args{1};output=args{2};mode=args{3};
snapshot=fullfile(output,'sources');addpath(snapshot);
data=load(fullfile(snapshot,'SimulatedData_SigmaSq_1.mat'));
if strcmp(mode,'probe')
 rand('state',20261004);randn('state',20261004);x=sqrt(5)*randn(512,1);beta=std(x)*.3;
 targetlog=-.5*(data.y(end)-x.^2/20).^2-.5*log(2*pi);
 target=exp(targetlog-max(targetlog));G=exp(-beta*(x-x').^2);
 tic;[w,rss,residual,exitflag]=lsqnonneg(G,target);seconds=toc;
 gradient=G'*(G*w-target);
 probe=struct('seconds',seconds,'rss',rss,'exitflag',exitflag,...
  'positive_coefficients',sum(w>0),'minimum_gradient',min(gradient),...
  'max_complementarity',max(abs(w.*gradient)),'beta',beta,...
  'seed',20261004,'particles',512,'scope','SINGLE_TERMINAL_NNLS_COST_ONLY');
 save('-mat7-binary',fullfile(output,'probe.mat'),'probe','x','target','w');return;
end
assert(strcmp(which('cSMC_Resample'),fullfile(snapshot,'cSMC_Resample.m')));
N=64;H=6;M=16;y=data.y(1:H);variance=1;
Initial=struct('Mean',0,'Var',5,'Precision',.2,'MeanPrecision',0);
Transition=struct('G',10,'Ginv',.1,'LogG',log(10));
Transition.Mean=@(t,x).5*x+25*x./(1+x.^2)+8*cos(1.2*t);
Observation=struct();Observation.LogDensity=@(t,x)-.5*(y(t)-x.^2/20).^2-.5*log(2*pi);
Parameters=struct('Particles',N,'Knots',M,'Time',H-1,'Iterations',1,...
 'KnotsType','particles','BandwidthFactor',.3);
seed=20261003;rand('state',seed);randn('state',seed);literal=cSMC_Resample(Parameters,Initial,Transition,Observation);
rand('state',seed);randn('state',seed);observed=cSMC_Resample_observed(Parameters,Initial,Transition,Observation);
rand('state',seed);randn('state',seed);corrected=cSMC_Resample_corrected(Parameters,Initial,Transition,Observation);
same=true;source_tv=0;corrected_tv=0;affected=0;rounds=cell(1,2);
for r=1:2
 same=same&&isequal(literal{r,2}.Trajectory,observed{r,2}.Trajectory)...
  &&isequal(literal{r,2}.LogNormConst,observed{r,2}.LogNormConst);
 tv=observed{r,2}.AuditComponentWeightsTV;
 source_tv=max(source_tv,max(tv(:)));affected=affected+sum(tv(:)>1e-12);
 tv=corrected{r,2}.AuditComponentWeightsTV;corrected_tv=max(corrected_tv,max(tv(:)));
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
  lo=previous(idx);hi=cumulative(idx);assert(all(hi>lo));
  component_uniform(t,:)=((lo+hi)/2)';
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
% Exact structural controls: distinct parents, nonidentity ancestry, two modes.
m=[-4;4];knots=[-4,4];beta=1;q=1;ancestor=[2;1];
lp=-beta/(1+2*beta*q)*(m-knots).^2;prob=exp(lp-max(lp,[],2));prob=prob./sum(prob,2);
mu=(m(ancestor)+2*beta*q*knots)/(1+2*beta*q);
wrong_mean=sum(prob.*mu,2);right_mean=sum(prob(ancestor,:).*mu,2);
identity_tv=max(sum(abs(prob-prob([1;2],:)),2))/2;
uniform_prob=ones(2,2)/2;uniform_tv=max(sum(abs(uniform_prob-uniform_prob(ancestor,:)),2))/2;
exact_mean_difference=max(abs(wrong_mean-right_mean));
assert(same && source_tv>1e-6 && corrected_tv==0 && exact_mean_difference>1);
assert(identity_tv==0 && uniform_tv==0);
fixture=struct('scope','CORRECTED_PAPER_MIXTURE_CONSUMER_DIAGNOSTIC','seed',seed,...
 'particles',N,'observations',H,'components',M,'initial_variance',5,...
 'transition_variance',10,'observation_variance',variance,'y',y,'rounds',{rounds});
summary=struct('status','PASS','scope','RBF_SOURCE_CONDITIONAL_KERNEL_AUDIT',...
 'replication','INCOMPLETE','source_verdict','WRONG_CONDITIONAL_KERNEL',...
 'instrumentation_bitwise_unchanged',same,'source_max_component_weights_tv',source_tv,...
 'source_affected_transitions',affected,'corrected_max_component_weights_tv',corrected_tv,...
 'analytic_wrong_mean_difference',exact_mean_difference,'identity_ancestry_tv',identity_tv,...
 'uniform_policy_tv',uniform_tv,'seed',seed,'cpu_only',true,'gpu_intentionally_hidden',true);
save('-mat7-binary',fullfile(output,'export.mat'),'fixture','summary','literal','observed','corrected');
