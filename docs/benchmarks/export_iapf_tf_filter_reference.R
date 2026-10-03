# Independent R reconstruction of the actual TF fixed-guide consumer.
# It deliberately matches its x0 propagation and every-step resampling,
# not the separate paper-study R adaptive-resampling consumer.
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1)
out<-args[1];dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
f32<-function(x)readBin(writeBin(x,raw(),size=4),"numeric",n=length(x),size=4)
writevec<-function(x,file)write.table(matrix(as.vector(x),ncol=1),file,sep=",",row.names=FALSE,col.names=FALSE)
for(d in c(1,2,5)) {
  N<-64;T<-5;o<-d;dir<-file.path(out,paste0("d",d));dir.create(dir)
  set.seed(93600000+d)
  theta<-c(.55,log(.6),log(.7),1,.2,log(.8))
  observations<-matrix(rnorm(T*o),T,o);initial<-matrix(rnorm(N*d),N,d)
  noise<-lapply(1:T,function(t)matrix(rnorm(N*d),N,d))
  uniforms<-matrix(runif((T+1)*N),T+1,N);mixture<-matrix(runif(T*N),T,N)
  centers<-matrix(rnorm(T*d)/2,T,d)
  covariances<-lapply(1:T,function(t)diag(seq(.4,.8,length.out=d),d)+matrix(.02,d,d))
  floors<-vapply(covariances,function(V)-.5*(d*log(2*pi)+2*sum(log(diag(chol(V)))))-3,0.)
  theta_model<-function(th) {
    A<-diag(th[1],d);if(d>1)A[cbind(1:(d-1),2:d)]<-f32(.08)
    list(A=A,Q=exp(2*th[2])*diag(seq(1,f32(1.3),length.out=d),d),
      R=exp(2*th[3])*diag(o),H=th[4]*(diag(1,o,d)+matrix(f32(.15),o,d)),
      mean=th[5]*seq(f32(.7),f32(1.1),length.out=d),
      P=exp(2*th[6])*(diag(d)+matrix(f32(.15),d,d)))
  }
  guides<-lapply(1:T,function(t)iapf_gaussian_twist(centers[t,],covariances[[t]],floors[t]))
  evaluate<-function(th,constant=FALSE) {
    m<-theta_model(th);x<-sweep(initial%*%chol(m$P),2,m$mean,"+")
    clouds<-list();labels<-branches<-integer();margins<-numeric()
    future<-function(x,t)if(constant)rep(0,N)else iapf_log_integral(x%*%t(m$A),m$Q,guides[[t]])
    resample<-function(logw,u) {
      w<-exp(logw-iapf_logsum(logw));cum<-cumsum(w)
      indices<-vapply(u,function(v)min(which(cum>v)),1L)
      margins<<-c(margins,min(abs(outer(cum,u,"-"))))
      list(indices=indices,term=iapf_logmean(logw))
    }
    rs<-resample(future(x,1),uniforms[1,]);total<-rs$term
    x<-x[rs$indices,,drop=FALSE];labels<-c(labels,rs$indices)
    for(t in 1:T) {
      mean<-x%*%t(m$A)
      if(constant) {
        x<-mean+noise[[t]]%*%chol(m$Q);lp<-rep(0,N)
      } else {
        p<-iapf_proposal(mean,m$Q,guides[[t]])
        prob<-exp(iapf_log_normal(mean,guides[[t]]$mean,m$Q+guides[[t]]$covariance)-p$log_integral)
        choice<-mixture[t,]<prob;branches<-c(branches,choice)
        margins<-c(margins,min(abs(mixture[t,]-prob)))
        x<-mean+noise[[t]]%*%chol(m$Q)
        adapted<-p$means+noise[[t]]%*%chol(p$covariance)
        x[choice,]<-adapted[choice,,drop=FALSE]
        lp<-iapf_log_twist(x,guides[[t]])
      }
      clouds[[t]]<-x
      lg<-iapf_log_normal(matrix(rep(observations[t,],each=N),N,o),x%*%t(m$H),m$R)
      lf<-if(t<T)future(x,t+1)else rep(0,N)
      rs<-resample(lg+lf-lp,uniforms[t+1,]);total<-total+rs$term
      x<-x[rs$indices,,drop=FALSE];labels<-c(labels,rs$indices)
    }
    list(value=total,clouds=clouds,labels=labels,branches=branches,min_margin=min(margins))
  }
  writevec(theta,file.path(dir,"theta.csv"));writevec(t(observations),file.path(dir,"observations.csv"))
  writevec(t(initial),file.path(dir,"initial.csv"));writevec(unlist(lapply(noise,t)),file.path(dir,"noise.csv"))
  writevec(t(uniforms),file.path(dir,"uniforms.csv"));writevec(t(mixture),file.path(dir,"mixture.csv"))
  writevec(t(centers),file.path(dir,"centers.csv"));writevec(unlist(lapply(covariances,t)),file.path(dir,"covariances.csv"))
  writevec(floors,file.path(dir,"floors.csv"))
  if(d==2)for(kind in c("fixture","sentinel")) {
    cs<-if(kind=="fixture")centers else matrix(c(0,0,1,-1,-2,2,3,-3,-4,4),T,d,byrow=TRUE)
    m<-theta_model(theta);means<-initial*.3
    expected<-lapply(1:T,function(t) {
      # Independent precision-form product, not the TF gain/subtraction formula.
      qi<-solve(m$Q);vi<-solve(covariances[[t]]);posterior<-solve(qi+vi)
      rhs<-means%*%qi+matrix(rep(as.vector(vi%*%cs[t,]),each=N),N,d)
      rhs%*%posterior+noise[[t]]%*%chol(posterior)
    })
    writevec(unlist(lapply(expected,t)),file.path(dir,paste0("center-probe-",kind,".csv")))
  }
  for(constant in c(FALSE,TRUE)) {
    base<-evaluate(theta,constant);fd<-numeric(6);fdhalf<-numeric(6)
    for(k in 1:6)for(h in c(1e-5,5e-6)) {
      delta<-rep(0,6);delta[k]<-h
      a<-evaluate(theta+delta,constant);b<-evaluate(theta-delta,constant)
      stopifnot(identical(a$labels,base$labels),identical(b$labels,base$labels),
        identical(a$branches,base$branches),identical(b$branches,base$branches))
      if(h==1e-5)fd[k]<-(a$value-b$value)/(2*h)else fdhalf[k]<-(a$value-b$value)/(2*h)
    }
    stopifnot(max(abs(fd-fdhalf))<1e-5)
    prefix<-if(constant)"constant"else "fitted"
    writevec(base$value,file.path(dir,paste0(prefix,"-value.csv")))
    writevec(unlist(lapply(base$clouds,t)),file.path(dir,paste0(prefix,"-clouds.csv")))
    writevec(fdhalf,file.path(dir,paste0(prefix,"-score.csv")))
    writevec(c(base$min_margin,max(abs(fd-fdhalf))),file.path(dir,paste0(prefix,"-margins.csv")))
  }
}
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
