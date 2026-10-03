# Independent CPU R diagnostics, not a production learner or author code.
# See iapf-renewed-mechanism-campaign-2026-09-22.md for the evidence contract.
iapf_diag_logdet <- function(M) 2*sum(log(diag(iapf_chol(M))))
iapf_diag_gaussian_kl <- function(m0,V0,m1,V1) {
  J <- chol2inv(iapf_chol(V1)); delta <- m1-m0
  .5*(sum(J*t(V0))+sum(delta*as.vector(J%*%delta))-length(m0)+
        iapf_diag_logdet(V1)-iapf_diag_logdet(V0))
}
iapf_diag_moments <- function(model) {
  # Independent forward/RTS moments, retaining pre-observation predictions.
  T <- model$horizon; d <- model$dimension
  a <- b <- vector("list",T); P <- V <- vector("list",T)
  for(t in seq_len(T)) {
    a[[t]] <- if(t==1)model$initial_mean else as.vector(model$A%*%b[[t-1]])
    P[[t]] <- if(t==1)model$initial_covariance else
      model$A%*%V[[t-1]]%*%t(model$A)+model$transition_covariance
    K <- P[[t]]%*%t(model$C)%*%
      solve(model$C%*%P[[t]]%*%t(model$C)+model$R)
    b[[t]] <- a[[t]]+as.vector(K%*%(model$observations[t,]-model$C%*%a[[t]]))
    H <- diag(d)-K%*%model$C
    V[[t]] <- H%*%P[[t]]%*%t(H)+K%*%model$R%*%t(K)
  }
  sm <- b; SV <- V
  if(T>1)for(t in (T-1):1) {
    G <- V[[t]]%*%t(model$A)%*%solve(P[[t+1]])
    sm[[t]] <- b[[t]]+as.vector(G%*%(sm[[t+1]]-a[[t+1]]))
    SV[[t]] <- V[[t]]+G%*%(SV[[t+1]]-P[[t+1]])%*%t(G)
    SV[[t]] <- .5*(SV[[t]]+t(SV[[t]]))
  }
  list(prediction_mean=a,prediction_covariance=P,
       smoothing_mean=sm,smoothing_covariance=SV)
}
iapf_diag_component <- function(a,P,guide) {
  if(guide$constant)return(list(mean=a,covariance=P,log_floor_probability=-Inf))
  proposal <- iapf_proposal(matrix(a,1),P,guide)
  list(mean=as.vector(proposal$means),covariance=proposal$covariance,
       log_floor_probability=guide$log_floor-proposal$log_integral[1])
}
iapf_diag_projection <- function(a,P,b,S,initial_precision) {
  J0 <- solve(P); d <- length(a)
  evaluate <- function(j) {
    J <- J0+diag(j,d); C <- chol2inv(iapf_chol(J))
    list(value=.5*(sum(J*t(S))-iapf_diag_logdet(J)),
         gradient=.5*(diag(S)-diag(C)),covariance=C)
  }
  fit <- optim(initial_precision,function(j)evaluate(j)$value,
    function(j)evaluate(j)$gradient,method="L-BFGS-B",lower=rep(0,d),
    control=list(maxit=1000,factr=10,pgtol=1e-11))
  ev <- evaluate(fit$par)
  projected <- ev$gradient
  active <- fit$par<=1e-12; projected[active] <- pmin(projected[active],0)
  # Newton polishing checks the convex optimum, not performance tuning.
  for(iter in seq_len(20)) {
    if(max(abs(projected))<=1e-10)break
    free <- which(!active | ev$gradient<0)
    step <- rep(0,d)
    step[free] <- -solve(.5*ev$covariance[free,free,drop=FALSE]^2,ev$gradient[free])
    alpha <- 1
    if(any(step<0))alpha <- min(alpha,.999*min(-fit$par[step<0]/step[step<0]))
    accepted <- FALSE
    for(bt in seq_len(30)) {
      jnew <- pmax(fit$par+alpha*step,0); enew <- evaluate(jnew)
      if(enew$value<=ev$value+1e-14) {accepted<-TRUE;break}
      alpha <- alpha/2
    }
    if(!accepted)break
    fit$par <- jnew; ev <- enew
    active <- fit$par<=1e-12; projected <- ev$gradient
    projected[active] <- pmin(projected[active],0)
  }
  kkt <- max(abs(projected))
  iapf_assert(is.finite(kkt) && kkt<=1e-7,"diagonal projection KKT failed")
  list(precision=fit$par,mean=b,covariance=ev$covariance,kkt=kkt,
       kl=iapf_diag_gaussian_kl(b,S,b,ev$covariance),active=sum(active))
}
iapf_diag_shape <- function(logratio) sqrt(mean((logratio-mean(logratio))^2))
iapf_diag_mixture <- function(x,a,P,b,S,guide) {
  component <- iapf_diag_component(a,P,guide)
  lp <- iapf_log_normal(x,b,S)
  lq <- iapf_log_normal(x,component$mean,component$covariance)
  if(guide$constant)lmix <- lq else {
    logalpha <- component$log_floor_probability
    logone <- log(-expm1(logalpha))
    lmix <- iapf_logadd(logone+lq,logalpha+iapf_log_normal(x,a,P))
  }
  z <- lp-lmix; floor_delta <- lq-lmix
  list(kl=mean(z),mcse=sd(z)/sqrt(length(z)),
       floor_kl_change=mean(floor_delta),floor_change_mcse=sd(floor_delta)/sqrt(length(z)),
       floor_probability=exp(component$log_floor_probability))
}
