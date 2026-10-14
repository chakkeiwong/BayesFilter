% Independent CPU fixed-cloud replay with exact healthy source controls.
args=argv(); root=args{1}; output=args{2}; baseline=args{3};
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst/distributions'));
snapshot=fullfile(output,'sources'); addpath(snapshot); cd(snapshot);
driver=fileread(fullfile(snapshot,'cSMC_VaryF.m'));
prefix=strsplit(driver,'% Vary F'); assert(numel(prefix)==2); eval(prefix{1});
assert(d==8 && T==100 && N==512 && I==4);
global LORENZ_REPLAY_AUDIT;
healthy_count=0; healthy_exact=true;
files=dir(fullfile(baseline,'author-F*.mat')); assert(numel(files)==6);
for file_index=1:numel(files)
  old=load(fullfile(baseline,files(file_index).name));
  for stage=1:(size(old.result,1)-1)
    LORENZ_REPLAY_AUDIT={};
    policy=author_lorenz_fit_replay(Parameters,Initial,Transition,old.result{stage,2},old.result{stage,1});
    assert(isequal(policy,old.result{stage+1,1}));
    healthy_count=healthy_count+1;
  end
end
assert(healthy_count==16);
data=load(fullfile(output,'input.mat'));
assert(isequal(data.initial_covariance,Initial.Cov));
assert(isequal(data.transition_covariance,Transition.G));
F=data.forcing; Lorenz=@(x) ((PT*x'-PP*x').*(P*x')-x'+F)';
Transition.Mean=@(t,x) RungeKutta4(x,interval,Lorenz);
candidate_valid=false(1,data.stage_count); failure_time=-ones(1,data.stage_count);
minimum_eigenvalue=zeros(1,data.stage_count); evaluated_times=zeros(1,data.stage_count);
candidate_traces=cell(1,data.stage_count); candidate_policies=cell(1,data.stage_count);
for stage=1:data.stage_count
  item=data.(sprintf('case%d',stage)); Policy=struct();
  Policy.A=item.A; Policy.b=item.b; Policy.c=item.c(:);
  Policy.G=zeros(T,d,d); Policy.LogDetG=zeros(T,1);
  SMC=struct(); SMC.Trajectory=permute(item.x,[2 3 1]);
  SMC.LogIncrementalWeight=item.log_incremental_weight';
  SMC.ShiftTransitionMean=zeros(N,d,T); SMC.QuadForm=zeros(N,T);
  for t=1:T
    precision=Transition.Ginv+2*squeeze(Policy.A(t+1,:,:));
    assert(min(eig(0.5*(precision+precision')))>0);
    G=inv(precision); Policy.G(t,:,:)=G; Policy.LogDetG(t)=log(det(G));
    mean=Transition.Mean(t,squeeze(SMC.Trajectory(:,:,t)));
    shift=bsxfun(@minus,mean*Transition.Ginv,Policy.b(t+1,:));
    SMC.ShiftTransitionMean(:,:,t)=shift;
    SMC.QuadForm(:,t)=0.5*sum((shift*G).*shift,2);
  end
  LORENZ_REPLAY_AUDIT={};
  try
    candidate_policies{stage}=author_lorenz_fit_replay(Parameters,Initial,Transition,SMC,Policy);
    candidate_valid(stage)=true;
  catch err
    if ~strcmp(err.identifier,'LorenzReplay:InvalidPrecision'); rethrow(err); end
    failure_time(stage)=LORENZ_REPLAY_AUDIT{end}.time;
  end
  candidate_traces{stage}=LORENZ_REPLAY_AUDIT;
  evaluated_times(stage)=numel(LORENZ_REPLAY_AUDIT);
  minimum_eigenvalue(stage)=min(cellfun(@(v) v.min_eigenvalue,LORENZ_REPLAY_AUDIT));
  fprintf('Fixed TF cloud row 0 refit %d: valid=%d failure_time=%d min_eigenvalue=%.17g\n',...
    stage,candidate_valid(stage),failure_time(stage),minimum_eigenvalue(stage));
end
octave_version=version(); x2fx_path=which('x2fx');
save('-mat7-binary',fullfile(output,'replay.mat'),'healthy_count','healthy_exact',...
 'candidate_valid','failure_time','minimum_eigenvalue','evaluated_times',...
 'candidate_traces','candidate_policies','octave_version','x2fx_path');
