source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_scale_aware_fit.R")
p <- readRDS("docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03/min_mass_Nminus1-9262001.rds")
inp <- p$inputs[[p$time]]
data <- readRDS("docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03/data-9262001.rds")
cat("data names:",paste(names(data),collapse=","),"\n")
model <- data$bundle$model
cat("bundle names:",paste(names(data$bundle),collapse=","),"\n")
cat("model names:",paste(names(model),collapse=","),"\n")
x <- inp$points
tw <- inp$next_twist
obs <- model$log_observation(x,model$observations[p$time,],p$time)
gauss <- obs + iapf_log_normal(model$transition_mean(x,p$time+1), tw$mean, model$transition_covariance+tw$covariance)
floor <- obs + tw$log_floor
mix <- inp$log_targets
branch <- exp(gauss - pmax(gauss,floor))
cat("target ranges mix/gauss/floor:",paste(range(mix),collapse=" "),paste(range(gauss),collapse=" "),paste(range(floor),collapse=" "),"\n")
cat("gaussian branch probability contribution min/median/max:", paste(signif(quantile(branch,c(0,.25,.5,.75,1)),5),collapse=" "),"\n")
cat("log mix-minus-gauss quantiles:",paste(signif(quantile(mix-gauss,c(0,.25,.5,.75,1)),5),collapse=" "),"\n")
cat("points mean/sd:",paste(signif(colMeans(x),5),collapse=" "),paste(signif(sqrt(colMeans(sweep(x,2,colMeans(x))^2)),5),collapse=" "),"\n")
cat("A:\n"); print(model$A)
