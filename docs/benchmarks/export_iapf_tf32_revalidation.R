# Independent CPU-only diagnostic data, guide and Kalman authority for phase18.
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args)==1)
out <- args[1]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
source("docs/benchmarks/reference_iapf_tf_adaptive.R")
writevec <- function(x,path) write.table(as.vector(x),path,sep=",",row.names=FALSE,col.names=FALSE)
readvec <- function(path) scan(path,quiet=TRUE)
f32 <- tfref_f32
roundmat <- function(x) matrix(f32(as.vector(x)),nrow(x),ncol(x))
theta <- c(.55,log(.6),log(.7),1,.2,log(.8))
spec <- data.frame(id=character(),d=integer(),N=integer(),T=integer(),data_seed=integer())
for(d in c(1,2,5)) {
  spec <- rbind(spec,data.frame(id=paste0("old_d",d),d=d,N=64,T=5,data_seed=93600000+d))
  for(j in 1:2) spec <- rbind(spec,data.frame(id=paste0("fresh",j,"_d",d),d=d,N=64,T=5,
    data_seed=2026092200+100*j+d))
}
for(j in 1:2) spec <- rbind(spec,data.frame(id=paste0("long",j,"_d5"),d=5,N=128,T=25,
  data_seed=2026092405+100*j))
rows <- checks <- list()
for(i in seq_len(nrow(spec))) {
  s <- spec[i,]; d <- s$d; T <- s$T; m <- tfref_model(f32(theta),d,d)
  old <- grepl("^old",s$id); path <- file.path(out,s$id); dir.create(path)
  if(old) {
    src <- file.path(out,"old",paste0("d",d))
    y <- matrix(readvec(file.path(src,"observations.csv")),T,d,byrow=TRUE)
    centers <- matrix(readvec(file.path(src,"centers.csv")),T,d,byrow=TRUE)
    vs <- readvec(file.path(src,"covariances.csv"))
    covs <- lapply(seq_len(T),function(t)matrix(vs[(t-1)*d*d+seq_len(d*d)],d,d,byrow=TRUE))
    floors <- readvec(file.path(src,"floors.csv"))
  } else {
    set.seed(s$data_seed)
    x <- as.vector(m$mean+t(chol(m$P))%*%rnorm(d)); y <- matrix(0,T,d)
    for(t in seq_len(T)) {
      x <- as.vector(m$A%*%x+t(chol(m$Q))%*%rnorm(d))
      y[t,] <- as.vector(m$H%*%x+t(chol(m$R))%*%rnorm(d))
    }
    centers <- matrix(rnorm(T*d)/2,T,d)
    covs <- lapply(seq_len(T),function(t)diag(seq(.4,.8,length.out=d),d)+matrix(.02,d,d))
    floors <- vapply(covs,function(V)-.5*(d*log(2*pi)+2*sum(log(diag(chol(V)))))-3,0.)
  }
  y <- roundmat(y)
  # R paper Kalman starts at x1: propagate the TF consumer's x0 once.
  pm <- iapf_linear_model(y,m$A,initial_mean=as.vector(m$A%*%m$mean),
    initial_covariance=m$A%*%m$P%*%t(m$A)+m$Q,
    Q=m$Q,C=m$H,R=m$R)
  oracle <- iapf_kalman(pm)$log_likelihood
  # Independent full observation covariance, without Kalman updates.
  means <- list(); covariances <- list(); mu <- m$mean; P <- m$P
  for(t in seq_len(T)) {
    mu <- as.vector(m$A%*%mu); P <- m$A%*%P%*%t(m$A)+m$Q
    means[[t]] <- as.vector(m$H%*%mu); covariances[[t]] <- P
  }
  joint <- matrix(0,T*d,T*d)
  for(a in seq_len(T)) for(b in a:T) {
    cross <- covariances[[a]]
    if(b>a)for(k in seq_len(b-a))cross <- cross%*%t(m$A)
    block <- m$H%*%cross%*%t(m$H)
    if(a==b)block <- block+m$R
    ia <- (a-1)*d+seq_len(d); ib <- (b-1)*d+seq_len(d)
    joint[ia,ib] <- block; joint[ib,ia] <- t(block)
  }
  joint_value <- iapf_log_normal(matrix(as.vector(t(y)),1),unlist(means),joint)
  stopifnot(abs(joint_value-oracle)<1e-9)
  checks[[i]] <- data.frame(cell=s$id,kalman=oracle,joint=joint_value,error=abs(joint_value-oracle))
  guides <- list(stress=lapply(seq_len(T),function(t)iapf_gaussian_twist(centers[t,],covs[[t]],floors[t])),
    constant=lapply(seq_len(T),function(t)iapf_gaussian_twist(rep(0,d),diag(d))),
    observation=iapf_observation_twists(pm),oracle=iapf_exact_twists(pm))
  writevec(f32(theta),file.path(path,"theta.csv")); writevec(t(y),file.path(path,"observations.csv"))
  for(arm in names(guides)) {
    gs <- guides[[arm]]
    writevec(f32(unlist(lapply(gs,function(g)g$mean))),file.path(path,paste0(arm,"-centers.csv")))
    writevec(f32(unlist(lapply(gs,function(g)as.vector(t(g$covariance))))),file.path(path,paste0(arm,"-covariances.csv")))
    writevec(f32(vapply(gs,function(g)g$log_floor,0.)),file.path(path,paste0(arm,"-floors.csv")))
    rows[[length(rows)+1]] <- data.frame(s,arm=arm,kalman=oracle,particle_seed=2030000000+i*10000)
  }
}
write.csv(do.call(rbind,rows),file.path(out,"cells.csv"),row.names=FALSE)
write.csv(do.call(rbind,checks),file.path(out,"oracle-checks.csv"),row.names=FALSE)
writeLines(c("CPU-only independent R reference; CUDA_VISIBLE_DEVICES=-1",capture.output(sessionInfo())),
  file.path(out,"session-info.txt"))
