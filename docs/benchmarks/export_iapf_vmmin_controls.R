# Independent R vmmin semantic controls; CPU reference only.
args<-commandArgs(TRUE);out<-args[2]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
options(digits=17)
write_matrix<-function(x,path) {
 x<-as.matrix(x);encoded<-matrix(sprintf("%.17g",as.numeric(x)),nrow=nrow(x),ncol=ncol(x))
 write.table(encoded,path,sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE)
}
problems<-list(
 quadratic=list(z=c(1,2),fn=function(z)sum(c(1,3)*(z-c(-.2,.4))^2),
  gr=function(z)2*c(1,3)*(z-c(-.2,.4)),maxit=100),
 overflow_trial=list(z=c(-10),fn=function(z)exp(z[1])-1000*z[1],
  gr=function(z)exp(z[1])-1000,maxit=100),
 rosenbrock=list(z=c(-1.2,1),fn=function(z)100*(z[2]-z[1]^2)^2+(1-z[1])^2,
  gr=function(z)c(-400*z[1]*(z[2]-z[1]^2)-2*(1-z[1]),200*(z[2]-z[1]^2)),maxit=100),
 stationary=list(z=c(2,-3),fn=function(z)0,gr=function(z)c(0,0),maxit=100),
 nochange=list(z=c(1e20),fn=function(z)z[1],gr=function(z)1,maxit=100),
 negative_curvature=list(z=c(1,-2),fn=function(z)-.5*sum(z^2),gr=function(z)-z,maxit=4),
 invalid_initial=list(z=c(1000),fn=function(z)exp(z[1]),gr=function(z)exp(z[1]),maxit=100))
index<-list()
for(name in names(problems)) {
 p<-problems[[name]];folder<-file.path(out,name);dir.create(folder)
 calls<-list();accepted<-list()
 fn<-function(z){value<-p$fn(z);calls[[length(calls)+1L]]<<-c(z,value);value}
 gr<-function(z){value<-p$gr(z);accepted[[length(accepted)+1L]]<<-c(z,p$fn(z),value);value}
 fit<-tryCatch(optim(p$z,fn,gr,method="BFGS",
  control=list(maxit=p$maxit,reltol=sqrt(.Machine$double.eps))),error=function(e)e)
 failed<-inherits(fit,"error")
 write_matrix(p$z,file.path(folder,"initial.csv"))
 if(length(calls)>0)write_matrix(do.call(rbind,calls),file.path(folder,"calls.csv"))
 if(length(accepted)>0)write_matrix(do.call(rbind,accepted),file.path(folder,"accepted.csv"))
 if(!failed)write_matrix(fit$par,file.path(folder,"endpoint.csv"))
 index[[length(index)+1L]]<-data.frame(name=name,dim=length(p$z),maxit=p$maxit,
  rejected_initial=failed,code=if(failed)-1 else fit$convergence,
  function_calls=if(failed)NA else fit$counts[1],gradient_calls=if(failed)NA else fit$counts[2],
  loss=if(failed)NA else fit$value,
  nonfinite_function_calls=sum(!vapply(calls,function(z)is.finite(tail(z,1)),logical(1))))
}
write.csv(do.call(rbind,index),file.path(out,"index.csv"),row.names=FALSE)
writeLines(c(R.version.string,"CPU reference; GPU intentionally hidden",
 "Explicit controls; not original iAPF-author fitting identity"),file.path(out,"runtime.txt"))
write.csv(data.frame(status="PASS",verdict="REFERENCE_CONTROLS_EXPORTED",
 count=length(problems)),file.path(out,"decision.csv"),row.names=FALSE)
