% Independent CPU diagnostic. Instrumentation only; exact source replay required.
args=argv(); root=args{1}; output=args{2}; baseline=args{3};
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst'));
addpath(fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst/distributions'));
snapshot=fullfile(output,'sources'); addpath(snapshot); cd(snapshot);
driver=fileread(fullfile(snapshot,'cSMC_VaryF.m'));
prefix=strsplit(driver,'% Vary F'); assert(numel(prefix)==2); eval(prefix{1});
assert(d==8 && T==100 && N==512 && I==4);
assert(strcmp(Parameters.Approximator,'quadratic') && strcmp(Parameters.Adaptive,'yes'));
instrumented=fullfile(output,'instrumented'); addpath(instrumented); cd(instrumented);
assert(strcmp(which('cSMC_Resample'),fullfile(instrumented,'cSMC_Resample.m')));
assert(strcmp(which('RungeKutta4'),fullfile(snapshot,'RungeKutta4.m')));
assert(strcmp(which('Systematic_Resampling'),fullfile(snapshot,'Systematic_Resampling.m')));
forcing_values=[2.5,4.8801,8.5]; seeds=[20260929,20260930];
global LORENZ_AUDIT_FITS;
for forcing_index=1:numel(forcing_values)
 F=forcing_values(forcing_index);
 Lorenz=@(x)((PT*x'-PP*x').*(P*x')-x'+F)';
 Transition.Mean=@(t,x)RungeKutta4(x,interval,Lorenz);
 for seed_index=1:numel(seeds)
  seed=seeds(seed_index); rand('state',seed);randn('state',seed);
  LORENZ_AUDIT_FITS={};
  result=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
  filename=sprintf('author-F%.4f-seed%d.mat',F,seed);
  original=load(fullfile(baseline,filename));
  assert(isequal(result,original.result));
  refinements=size(result,1)-1;
  assert(numel(LORENZ_AUDIT_FITS)==refinements*(T+1));
  for r=1:refinements
   for t=(T+1):-1:1
    item=LORENZ_AUDIT_FITS{(r-1)*(T+1)+(T+2-t)};
    assert(item.time==t);
    assert(isequal(item.X,result{r,2}.Trajectory(:,:,t)));
    assert(all(isfinite([item.X(:);item.Value(:);item.BetaHat(:);item.design_edges(:)])));
   end
  end
  captured=LORENZ_AUDIT_FITS; exact_source_replay=true;
  save('-mat7-binary',fullfile(output,filename),'captured','F','seed','refinements','exact_source_replay');
  fprintf('CAPTURE F=%.4f seed=%d refinements=%d exact_source_replay=1\n',F,seed,refinements);
 end
end
