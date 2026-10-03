# Deterministic continuous-risk calculation on already captured failed fits.
args <- commandArgs(TRUE); root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_support_diagnostics.R"))
stopifnot(!dir.exists(output)); dir.create(output,recursive=TRUE)
rows <- list()
for(id in c("lbfgs_d20_t92","nlminb_d20_t68")) {
  failed <- readRDS(file.path(root,"docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01",
    "attempt10-code-audit/results",paste0(id,"-failure.rds")))
  model <- failed$model; time <- failed$backward_time; d <- model$dimension
  reference <- iapf_linear_smoothing_marginal(model,time)
  target <- iapf_backward_target_mixture(model,time,failed$next_twist)
  candidates <- list(failed_fit=failed$parameters,initial_fit=failed$initial_parameters,
    diagonal_target_moments=c(target$mean,log(diag(target$covariance))),constant=NULL)
  for(name in names(candidates))for(component in c("gaussian","gaussian_plus_floor")) {
    parameters <- candidates[[name]]
    candidate <- if(name=="constant")iapf_constant_twist() else {
      lv <- parameters[d+seq_len(d)]
      floor <- if(component=="gaussian")-Inf else -.5*(d*log(2*pi)+sum(lv))-
        .5*qchisq(failed$N^(-failed$floor_tail_power),df=d,lower.tail=FALSE)
      iapf_gaussian_twist(parameters[seq_len(d)],diag(exp(lv),d),floor)
    }
    rows[[length(rows)+1]] <- data.frame(case=id,time=time,candidate=name,component=component,
      as.data.frame(iapf_exact_support_risk(reference,candidate,target)))
  }
}
write.csv(do.call(rbind,rows),file.path(output,"exact-risk.csv"),row.names=FALSE)
cat("COMPLETE",length(rows),"analytic continuous-risk calculations\n")
