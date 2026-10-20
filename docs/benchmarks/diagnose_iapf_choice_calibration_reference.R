# Independent FP64 CPU reference diagnostics, never a runtime/default route.
args <- commandArgs(TRUE)
root <- args[1]; refroot <- args[2]; out <- args[3]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",dir.exists(out))
source(file.path(refroot,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(refroot,"docs/benchmarks/reference_iapf_eq15_solvers.R"))
started <- proc.time()[3]
budget <- 850
check_time <- function() if(proc.time()[3]-started>budget)stop("CALIBRATION_BUDGET_LIMIT")
append_row <- function(row,name) {
  path <- file.path(out,name)
  write.table(row,path,sep=",",row.names=FALSE,col.names=!file.exists(path),
              append=file.exists(path),quote=TRUE,na="")
}
logdiag <- function(x,p) {
  d<-ncol(x); z<-sweep(x,2,p[seq_len(d)])
  -.5*(d*log(2*pi)+sum(p[d+seq_len(d)])+
    rowSums(sweep(z*z,2,exp(p[d+seq_len(d)]),"/")))
}
moments <- function(x,y) {
  w<-exp(y-max(y));w<-w/sum(w);m<-colSums(x*w)
  v<-colSums(sweep(x,2,m)^2*w)
  stopifnot(all(is.finite(v)),all(v>0))
  c(m,log(v))
}
shape <- function(p,x,y) {
  lp<-logdiag(x,p); a<-exp(lp-max(lp));b<-exp(y-max(y))
  sum((a-sum(a*b)/sum(b*b)*b)^2)/sum(a*a)
}
make_eval <- function(x,y,anchor,amplitude="profiled",scaled=TRUE) {
  d<-ncol(x); lp<-logdiag(x,anchor); fixed<-max(lp)
  b<-exp(y-max(y)); bn<-sum(b*b)
  normalization<-mean(exp(lp-fixed)^2)
  coords<-if(scaled)c(sqrt(exp(anchor[d+seq_len(d)])),rep(1,d)) else rep(1,2*d)
  if(amplitude=="joint")coords<-c(coords,1)
  calc<-function(p,needJ=TRUE) {
    m<-p[seq_len(d)];lv<-p[d+seq_len(d)];v<-exp(lv)
    z<-sweep(x,2,m);sq<-sweep(z*z,2,v,"/")
    a<-exp(-.5*(d*log(2*pi)+sum(lv)+rowSums(sq))-fixed)
    lam<-if(amplitude=="joint")exp(p[2*d+1]) else sum(a*b)/bn
    res<-(a-lam*b)/sqrt(nrow(x)*normalization)
    stopifnot(all(is.finite(res)))
    Jp<-cbind(sweep(z,2,v,"/"),.5*(sq-1))*a
    J<-if(amplitude=="joint")cbind(Jp,-lam*b) else
      Jp-outer(b,colSums(Jp*b)/bn)
    J<-sweep(J,2,coords,"*")/sqrt(nrow(x)*normalization)
    list(value=sum(res*res),gradient=2*colSums(J*res),J=J,residual=res,
         lambda=lam,max_log_density=max(logdiag(x,p[seq_len(2*d)])))
  }
  list(calc=calc,coords=coords,fixed=fixed,normalization=normalization,
    lambda=function(p)sum(exp(logdiag(x,p)-fixed)*b)/bn)
}
solve_fit <- function(x,y,initial,anchor,amplitude,solver,maxit=200,
                      radius=Inf,loose=FALSE,scaled=TRUE) {
  d<-ncol(x); eng<-make_eval(x,y,anchor,amplitude,scaled)
  p0<-if(amplitude=="joint")c(initial,log(eng$lambda(initial)))else initial
  coords<-eng$coords; start<-rep(0,length(p0));calls<-0L; trajectory<-list()
  numeric_limit<--log(.Machine$double.eps)
  low<-c(rep(-Inf,d),rep(-numeric_limit,d))
  high<-c(rep(Inf,d),rep(numeric_limit,d))
  if(amplitude=="joint"){low<-c(low,-600);high<-c(high,600)}
  lower<-(low-p0)/coords;upper<-(high-p0)/coords
  if(is.finite(radius)){lower<-pmax(lower,-radius);upper<-pmin(upper,radius)}
  evaluate<-function(u) {
    check_time();p<-p0+coords*u;ev<-eng$calc(p);calls<<-calls+1L
    trajectory[[length(trajectory)+1L]]<<-data.frame(call=calls,loss=ev$value,
      gradient=max(abs(ev$gradient)),shape=shape(p[seq_len(2*d)],x,y),
      logdet_change=sum(p[d+seq_len(d)]-initial[d+seq_len(d)]),
      logdensity_change=ev$max_log_density-max(logdiag(x,initial)),lambda=ev$lambda)
    ev
  }
  ev<-evaluate(start);u<-start; status<-"max_iterations";nit<-0L
  if(shape(initial,x,y)<1e-26) status<-"exact_start" else if(solver=="nlminb") {
    cache_u<-NULL;cache_ev<-NULL
    cached<-function(u) {
      if(is.null(cache_u)||!identical(u,cache_u)){cache_ev<<-evaluate(u);cache_u<<-u}
      cache_ev
    }
    fit<-nlminb(u,function(a)cached(a)$value,
      gradient=function(a)cached(a)$gradient,lower=lower,upper=upper,
      control=list(iter.max=maxit,eval.max=maxit*4,
        rel.tol=if(loose)1e-5 else 1e-10,x.tol=if(loose)1e-5 else 1e-10))
    u<-fit$par;nit<-fit$iterations
    status<-if(fit$convergence==0)"converged" else paste0("optimizer_",fit$convergence)
  } else {
    damp<-1e-3;nu<-2
    for(i in seq_len(maxit)) {
      nit<-i
      if(max(abs(ev$gradient))<=1e-8){status<-"converged";break}
      H<-crossprod(ev$J);g<-ev$gradient/2
      D<-pmax(diag(H),.Machine$double.eps*max(1,diag(H)))
      step<-tryCatch(as.vector(solve(H+diag(damp*D,length(D)),-g)),error=function(e)NULL)
      if(is.null(step)){status<-"linear_solve_failure";break}
      proposed<-u+step
      if(any(proposed<lower|proposed>upper))rho<--Inf else {
        next_ev<-tryCatch(evaluate(proposed),error=function(e)NULL)
        predicted<--2*sum(g*step)-sum(step*as.vector(H%*%step))
        rho<-if(is.null(next_ev)||predicted<=0)-Inf else (ev$value-next_ev$value)/predicted
      }
      if(is.finite(rho)&&rho>0) {
        u<-proposed;ev<-next_ev;damp<-damp*max(1/3,1-(2*rho-1)^3);nu<-2
      } else {damp<-damp*nu;nu<-2*nu}
      if(!is.finite(damp)||damp>1/.Machine$double.eps){status<-"trust_model_failure";break}
    }
  }
  p<-p0+coords*u;final<-eng$calc(p);pd<-p[seq_len(2*d)]
  bound<-any(abs(pd[d+seq_len(d)])>=numeric_limit-1e-5)
  local_boundary<-is.finite(radius)&&any(abs(u)>=radius-1e-5)
  rel<-shape(pd,x,y);ldchange<-sum(pd[d+seq_len(d)]-initial[d+seq_len(d)])
  density_change<-max(logdiag(x,pd))-max(logdiag(x,initial))
  # Explicit diagnostic flags, not sufficient conditions for scientific validity.
  escape<-density_change < -20 && rel>1e-6
  underflow<-final$value==0 && rel>1e-24
  list(parameters=pd,status=status,iterations=nit,calls=calls,loss=final$value,
    shape=rel,gradient=max(abs(final$gradient)),boundary=bound,
    local_boundary=local_boundary,escape=escape,underflow=underflow,
    logdet_change=ldchange,logdensity_change=density_change,
    trajectory=do.call(rbind,trajectory))
}
make_fixture <- function(d,kind,seed,heldseed) {
  n<-max(400L,6L*(2L*d+1L));nh<-max(1200L,20L*d)
  m<-rep(.2,d);v<-seq(.65,1.15,length.out=d)
  V<-if(kind=="correlated")outer(seq_len(d),seq_len(d),function(i,j).65^abs(i-j)) else diag(v,d)
  target<-if(kind %in% c("exact","correlated"))
    function(x)iapf_log_normal(x,m,V) else if(kind=="broad_tail")
    function(x)iapf_logadd(log(.9)+dnorm(x[,1],0,.7,log=TRUE),
                         log(.1)+dnorm(x[,1],0,3,log=TRUE)) else
    function(x)iapf_logadd(log(.5)+dnorm(x[,1],-1.5,.8,log=TRUE),
                         log(.5)+dnorm(x[,1],1.5,.8,log=TRUE))
  set.seed(seed);x<-matrix(rnorm(n*d),ncol=d)
  set.seed(heldseed);hx<-matrix(rnorm(nh*d),ncol=d)
  lap<-if(kind %in% c("exact","correlated"))c(m,log(1/diag(solve(V))))else if(kind=="broad_tail")
    c(0,log((.9/.7+.1/3)/(.9/.7^3+.1/3^3))) else NULL
  list(id=paste(kind,d,sep="_"),kind=kind,d=d,x=x,y=target(x),
    hx=hx,hy=target(hx),truth=if(kind=="exact")c(m,log(v))else NULL,
    laplace=lap,seeds=c(seed,heldseed))
}
# Stage 1, checked independently before any diagnostic comparisons.
set.seed(9250101);x<-matrix(rnorm(180),ncol=2);y<--.5*rowSums((x-.3)^2/.8)
anchor<-c(.1,-.1,log(c(.9,1.2)))
controls<-list()
for(am in c("joint","profiled")) {
  e<-make_eval(x,y,anchor,am)
  p<-if(am=="joint")c(anchor,log(e$lambda(anchor)))else anchor
  a<-e$calc(p);h<-1e-5
  fd<-vapply(seq_along(p),function(j){
    q<-p;q[j]<-q[j]+h*e$coords[j];r<-p;r[j]<-r[j]-h*e$coords[j]
    (e$calc(q)$value-e$calc(r)$value)/(2*h)},numeric(1))
  err<-max(abs(fd-a$gradient))/max(1,max(abs(fd)))
  stopifnot(err<1e-7)
  controls[[am]]<-data.frame(control=paste0(am,"_gradient"),error=err,pass=TRUE)
}
ej<-make_eval(x,y,anchor,"joint");ep<-make_eval(x,y,anchor,"profiled")
jv<-ej$calc(c(anchor,log(ej$lambda(anchor))))$value;pv<-ep$calc(anchor)$value
stopifnot(abs(jv-pv)<1e-12)
ep2<-make_eval(x,y+100,anchor,"profiled")
stopifnot(abs(ep$calc(anchor)$value-ep2$calc(anchor)$value)<1e-12)
tw<-iapf_gaussian_twist(.3,matrix(.8,1),log(1e-4))
numerical<-integrate(function(z)dnorm(z,.1,1.2)*exp(iapf_log_twist(matrix(z,ncol=1),tw)),
                     -Inf,Inf,rel.tol=1e-10)$value
exact<-exp(iapf_log_integral(matrix(.1,1),matrix(1.2^2,1),tw))
stopifnot(abs(numerical-exact)<1e-9)
write.csv(do.call(rbind,controls),file.path(out,"controls.csv"),row.names=FALSE)
writeLines("PASS: objective/profile, analytic gradients, target scaling and proposal normalizer",
           file.path(out,"controls.txt"))
cat("CONTROLS PASS\n");flush.console()
fixtures<-list();case<-0L
for(kind in c("exact","correlated","broad_tail","bimodal")) {
  dims<-switch(kind,exact=c(1L,5L,20L,80L),correlated=c(5L,20L,80L),1L)
  for(d in dims){case<-case+1L;fixtures[[length(fixtures)+1L]]<-
    make_fixture(d,kind,9251000+10*d+case,9252000+10*d+case)}
}
ids<-read.csv(file.path(refroot,"docs/plans/artifacts/iapf-kdm-replication-20260924-02/stage0/attempt02/input-identities.csv"))
ids<-ids[!duplicated(ids$input),]
stopifnot(nrow(ids)==10)
for(i in seq_len(nrow(ids))) {
  s<-readRDS(ids$path[i])
  fixtures[[length(fixtures)+1L]]<-list(id=paste0("saved_",ids$input[i]),kind="saved",
    d=ncol(s$points),x=s$points,y=s$log_targets,hx=NULL,hy=NULL,truth=NULL,laplace=NULL,
    seeds=NA,source=ids$path[i])
}
saveRDS(fixtures,file.path(out,"fixtures.rds"))
arms<-expand.grid(amplitude=c("profiled","joint"),solver=c("nlminb","lm"),
                  initialization=c("qr","moments"),stringsAsFactors=FALSE)
arms$maxit<-200L;arms$radius<-Inf;arms$loose<-FALSE;arms$scaled<-TRUE
arms$name<-apply(arms[,1:3],1,paste,collapse="_")
for(n in c("long","loose","unscaled","box025","box1","box4")) {
  r<-arms[1,];r$name<-n
  if(n=="long")r$maxit<-600L
  if(n=="loose")r$loose<-TRUE
  if(n=="unscaled")r$scaled<-FALSE
  if(n %in% c("box025","box1","box4"))r$radius<-c(box025=.25,box1=1,box4=4)[n]
  arms<-rbind(arms,r)
}
write.csv(arms,file.path(out,"arms.csv"),row.names=FALSE)
dir.create(file.path(out,"trajectories"))
for(f in fixtures) {
  check_time()
  starts<-list(qr=tryCatch(iapf_fit_initial(f$x,f$y,TRUE)$parameters,error=function(e)NULL),
               moments=moments(f$x,f$y))
  baseline<-c(starts,list(laplace=f$laplace))
  best<-Inf
  for(n in names(baseline))if(!is.null(baseline[[n]])) {
    tr<-shape(baseline[[n]],f$x,f$y)
    held<-if(is.null(f$hx))NA_real_ else shape(baseline[[n]],f$hx,f$hy)
    append_row(data.frame(fixture=f$id,kind=f$kind,d=f$d,baseline=n,
      train_shape=tr,held_shape=held),"baselines.csv")
    if(is.finite(held))best<-min(best,held)
  }
  anchor<-if(!is.null(starts$qr))starts$qr else starts$moments
  for(k in seq_len(nrow(arms))) {
    check_time();a<-arms[k,];ini<-starts[[a$initialization]];t0<-proc.time()[3]
    fit<-if(is.null(ini))structure(list(message="invalid_qr_start"),class=c("simpleError","error","condition")) else
      tryCatch(solve_fit(f$x,f$y,ini,anchor,a$amplitude,a$solver,a$maxit,
                        a$radius,a$loose,a$scaled),error=function(e)e)
    row<-data.frame(fixture=f$id,kind=f$kind,d=f$d,arm=a$name,status="error",
      iterations=NA_integer_,calls=NA_integer_,loss=NA_real_,shape=NA_real_,
      held_shape=NA_real_,best_baseline=if(is.finite(best))best else NA_real_,
      truth_error=NA_real_,gradient=NA_real_,boundary=FALSE,local_boundary=FALSE,
      escape=FALSE,underflow=FALSE,logdet_change=NA_real_,logdensity_change=NA_real_,
      numerical_pass=FALSE,nomination_cell=FALSE,wall_seconds=proc.time()[3]-t0,
      error="",stringsAsFactors=FALSE)
    if(inherits(fit,"error"))row$error<-conditionMessage(fit) else {
      for(n in c("status","iterations","calls","loss","shape","gradient","boundary",
                 "local_boundary","escape","underflow","logdet_change","logdensity_change"))
        row[[n]]<-fit[[n]]
      row$held_shape<-if(is.null(f$hx))NA_real_ else shape(fit$parameters,f$hx,f$hy)
      row$truth_error<-if(is.null(f$truth))NA_real_ else max(abs(fit$parameters-f$truth))
      row$numerical_pass<-fit$status%in%c("exact_start","converged")&&
        !any(c(fit$boundary,fit$local_boundary,fit$escape,fit$underflow))
      exactpass<-is.null(f$truth)||(row$shape<=1e-12&&row$truth_error<=1e-5)
      row$nomination_cell<-row$numerical_pass&&exactpass&&!is.na(row$held_shape)&&
        row$held_shape<=best+1e-8
      write.csv(fit$trajectory,file.path(out,"trajectories",paste0(f$id,"-",a$name,".csv")),row.names=FALSE)
    }
    row$wall_seconds<-proc.time()[3]-t0
    append_row(row,"fits.csv")
  }
  cat("FIXTURE",f$id,"complete\n");flush.console()
}
# Stage 4 is independent of solver nomination.
logsum2<-function(a,b){m<-pmax(a,b);v<-m+log(exp(a-m)+exp(b-m));v[is.infinite(m)&m<0]<--Inf;v}
second_moment<-function(mu,fm,fv,lf,delta_floor=FALSE) {
  true_m<-mu/2;true_v<-.5;fit_v<-fv/(1+fv);fit_m<-(fv*mu+fm)/(1+fv)
  z<-dnorm(mu,fm,sqrt(1+fv),log=TRUE);den<-logsum2(z,lf)
  lpi<-lf-den;lone<-z-den
  if(is.infinite(lf)&&lf<0&&2/true_v-1/fit_v<=0)
    return(list(value=Inf,error=NA_real_,message="analytically_divergent"))
  fun<-function(xx) {
    lt<-dnorm(xx,true_m,sqrt(true_v),log=TRUE)
    lq<-logsum2(lone+dnorm(xx,fit_m,sqrt(fit_v),log=TRUE),
                lpi+dnorm(xx,mu,1,log=TRUE))
    a<-exp(2*lt-lq);a[!is.finite(xx)]<-0;a
  }
  # Split around both means so an integrator cannot skip a narrow component.
  breaks<-sort(unique(c(-Inf,true_m,fit_m,mu,Inf)))
  vals<-lapply(seq_len(length(breaks)-1L),function(i)
    integrate(fun,breaks[i],breaks[i+1L],rel.tol=1e-8,subdivisions=1000L,stop.on.error=FALSE))
  list(value=sum(vapply(vals,function(v)v$value,numeric(1))),
       error=sum(vapply(vals,function(v)v$abs.error,numeric(1))),
       message=paste(unique(vapply(vals,function(v)v$message,character(1))),collapse=";"))
}
policies<-c("zero","tail2","tail4","tail8","peak2","peak4","peak8","mix1","mix2")
for(d in c(1,5,20,80))for(regime in c("healthy","underdispersed","miscentered"))
 for(offset in c(0,1,3))for(policy in policies) {
  check_time();N<-1000
  fv<-if(regime=="underdispersed").15 else 1
  fm<-if(regime=="miscentered")2 else 0
  peak<--.5*d*log(2*pi*fv);zmax<--.5*d*log(2*pi*(1+fv))
  lf<-switch(substr(policy,1,3),
    zer=-Inf,
    tai=peak-.5*qchisq(N^(-as.numeric(substr(policy,5,5))),df=d,lower.tail=FALSE),
    pea=peak-as.numeric(substr(policy,5,5))*log(N),
    mix={de<-N^(-as.numeric(substr(policy,4,4)));zmax+log(de)-log1p(-de)})
  # Offsets are measured from the fitted center, in process-sd units.
  mu<-fm+offset
  z<-zmax-.5*d*offset^2/(1+fv);lpi<-lf-logsum2(z,lf);pi0<-exp(lpi)
  center_pi<-exp(lf-logsum2(zmax,lf))
  res<-if(d==1)second_moment(mu,fm,fv,lf)else list(value=NA_real_,error=NA_real_,message="not_evaluated_high_dimension")
  append_row(data.frame(d=d,regime=regime,offset=offset,policy=policy,N=N,
    variance=fv,fit_mean=fm,parent_mean=mu,log_floor=lf,base_probability=pi0,
    center_probability=center_pi,center_nonharm=center_pi<=1/N+1e-14,
    healthy_second_moment_upper=if(regime=="healthy")1/(1-pi0)else NA_real_,
    second_moment=res$value,quadrature_error=res$error,quadrature_message=res$message),
    "floors.csv")
 }
cat("FLOORS complete\n");flush.console()
# Independent Gaussian comparator moments for each one-dimensional regime.
gauss_second<-function(tm,tv,qm,qv) {
  A<-1/tv-1/(2*qv);B<-2*tm/tv-qm/qv;C<--tm^2/tv+qm^2/(2*qv)
  if(A<=0)return(Inf)
  exp(.5*log(qv)-log(tv)-.5*log(2*A)+C+B*B/(4*A))
}
for(regime in c("healthy","underdispersed","miscentered"))for(offset in c(0,1,3)) {
  fv<-if(regime=="underdispersed").15 else 1;fm<-if(regime=="miscentered")2 else 0
  mu<-fm+offset;tm<-mu/2;tv<-.5
  bases<-list(base_transition=c(mu,1),unfloored_fit=c((fv*mu+fm)/(1+fv),fv/(1+fv)),
              observation_only=c(0,1),exact_conditional=c(tm,tv))
  for(n in names(bases)) {
    value<-gauss_second(tm,tv,bases[[n]][1],bases[[n]][2])
    append_row(data.frame(regime=regime,offset=offset,method=n,
                         second_moment=value),"floor-heuristics.csv")
  }
}
fs<-read.csv(file.path(out,"fits.csv"))
selection<-do.call(rbind,lapply(arms$name,function(a) {
  r<-fs[fs$arm==a,]
  generated<-r[r$kind!="saved",]
  saved<-r[r$kind=="saved",]
  data.frame(arm=a,generated_pass=sum(generated$nomination_cell),generated_total=nrow(generated),
    saved_numeric_pass=sum(saved$numerical_pass),saved_total=nrow(saved),
    eligible=all(generated$nomination_cell)&&all(saved$numerical_pass))
}))
write.csv(selection,file.path(out,"selection.csv"),row.names=FALSE)
saveRDS(list(controls=controls,selection=selection,wall_seconds=proc.time()[3]-started),
        file.path(out,"summary.rds"))
writeLines(c("stage=CALIBRATION_COMPLETE",paste0("nominated=",sum(selection$eligible)),
  "scientific_replication=INCOMPLETE","backend=independent_R_FP64_CPU_reference"),
  file.path(out,"terminal.txt"))
cat("TERMINAL nominees",sum(selection$eligible),"; elapsed",proc.time()[3]-started,"\n")
