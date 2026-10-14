% CPU-only source execution and diagnostic extraction, never a TF runtime.
args=argv(); root=args{1}; output=args{2};
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst/distributions'));
author=fullfile(root,'.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz');
addpath(author); snapshot=fullfile(output,'sources'); addpath(snapshot);
assert(strcmp(which('cSMC_Resample'),fullfile(snapshot,'cSMC_Resample.m')));
assert(strcmp(which('RungeKutta4'),fullfile(snapshot,'RungeKutta4.m')));
load(fullfile(author,'SimulatedData.mat'));
d=8;H=size(y,1);N=512;p=d-2;assert(H==101 && size(y,2)==6);
P=diag(ones(1,d-1),-1);P(1,d)=1;PT=P';PP=P*P;
drift=@(x)((PT*x'-PP*x').*(P*x')-x'+4.8801)';
Initial=struct('Mean',zeros(1,d),'Cov',.01*eye(d),'Precision',100*eye(d),...
 'MeanPrecision',zeros(1,d),'LogDetCov',d*log(.01),'QuadForm',0);
Transition=struct('G',.001*eye(d),'Ginv',1000*eye(d),'LogDetG',d*log(.001));
Transition.Mean=@(t,x)RungeKutta4(x,.1,drift);
obsvar=1e-4;Observation=struct('y',y);
Observation.LogDensity=@(t,x)-.5/obsvar*sum(bsxfun(@minus,y(t,:),x(:,1:p)).^2,2);
projection=[eye(p),zeros(p,d-p)];A=.5/obsvar*(projection'*projection);
G=inv(Transition.Ginv+2*A);
policy=struct('A',zeros(H,d,d),'b',-y*projection/obsvar,'c',.5/obsvar*sum(y.^2,2),...
 'G',zeros(H-1,d,d),'LogDetG',ones(H-1,1)*log(det(G)));
for t=1:H,policy.A(t,:,:)=A;end
for t=1:H-1,policy.G(t,:,:)=G;end
Parameters=struct('Particles',N,'Dimension',d,'Time',H-1,'Iterations',1,...
 'Approximator','quadratic','Adaptive','no');
seeds=[20260929,20260930];fixtures=cell(1,2);archive=cell(1,2);diagnostics=cell(1,2);started=tic;
for iseed=1:2
 rand('state',seeds(iseed));randn('state',seeds(iseed));
 result=cSMC_Resample(Parameters,Initial,Transition,Observation,policy);archive{iseed}=result;
 rounds=cell(1,2);summaries=cell(1,2);
 for stage=1:2
  pol=result{stage,1};smc=result{stage,2};noise=zeros(H,N,d);uniform=zeros(H,1);
  ratios=zeros(H,1);qr_ratios=zeros(H,1);precision_margins=zeros(H,1);
  for t=1:H
   X=smc.Trajectory(:,:,t);
   if t==1
    base_precision=Initial.Precision;base_mean=zeros(N,d);
   else
    base_precision=Transition.Ginv;
    base_mean=Transition.Mean(t-1,smc.Trajectory(smc.Ancestry(:,t-1),:,t-1));
   end
   prec=base_precision+2*squeeze(pol.A(t,:,:));cov=inv(prec);
   cm=bsxfun(@minus,base_mean*base_precision,pol.b(t,:))*cov;
   noise(t,:,:)=(X-cm)/chol(cov);
   lw=smc.LogIncrementalWeight(:,t);w=exp(lw-max(lw));w=w/sum(w);
   cdf=cumsum(w);cdf(end)=1;previous=[0;cdf(1:end-1)];idx=smc.Ancestry(:,t);
   lo=max([0;N*previous(idx)-(0:N-1)']);hi=min([1;N*cdf(idx)-(0:N-1)']);
   assert(hi>lo);uniform(t)=(lo+hi)/2;
   design=x2fx(X,'quadratic');ratios(t)=rcond(design'*design);
   centered=bsxfun(@rdivide,bsxfun(@minus,X,mean(X,1)),std(X,1,1));
   zd=x2fx(centered,'quadratic');qr_ratios(t)=rcond(zd'*zd);
   eigenvalues=eig(prec);precision_margins(t)=min(eigenvalues)/max(eigenvalues);
  end
  assert(all(isfinite(smc.LogNormConst)) && all(precision_margins>0));
  rounds{stage}=struct('A',pol.A,'b',pol.b,'c',pol.c,...
   'x',permute(smc.Trajectory,[3,1,2]),'ancestors',smc.Ancestry'-1,...
   'log_incremental_weight',smc.LogIncrementalWeight','logZ',smc.LogNormConst',...
   'ess',smc.ESS','normal',noise,'uniform',uniform,...
   'transition_means',permute(reshape(Transition.Mean(1,reshape(permute(smc.Trajectory,[1,3,2]),N*H,d)),N,H,d),[2,1,3]));
  summaries{stage}=struct('minimum_raw_gram_rcond',min(ratios),...
   'minimum_standardized_gram_rcond',min(qr_ratios),'minimum_precision_margin',min(precision_margins),...
   'logZ',smc.LogNormConst(end),'minimum_ess',min(smc.ESS));
 end
 fixtures{iseed}=struct('seed',seeds(iseed),'rounds',{rounds});diagnostics{iseed}=summaries;
end
fixture=struct('y',y,'dimension',d,'particles',N,'horizon',H,'forcing',4.8801,...
 'observation_variance',obsvar,'initial_covariance',Initial.Cov,'transition_covariance',Transition.G,...
 'replicates',{fixtures});
summary=struct('status','PASS','scope','LORENZ_SOURCE_CONSUMER_AND_FITTING_DIAGNOSTIC',...
 'replication','INCOMPLETE','cpu_only',true,'gpu_intentionally_hidden',true,...
 'seeds',seeds,'algorithm_seconds',toc(started),'diagnostics',{diagnostics},...
 'author_entry',which('cSMC_Resample'),'rk4_entry',which('RungeKutta4'));
save('-mat7-binary',fullfile(output,'export.mat'),'summary','fixture');
save('-binary',fullfile(output,'author-results.octave'),'archive','Parameters','y');
