p <- readRDS("docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03/min_mass_Nminus1-9262001.rds")
cat("payload names:", paste(names(p), collapse=","), "\n")
fit <- p$fit
cat("fit names:", paste(names(fit), collapse=","), "\n")
tr <- fit$trace
cat("trace dim:", paste(dim(tr), collapse="x"), "\n")
idx <- unique(c(seq_len(min(12,nrow(tr))), seq(20,nrow(tr),by=20), max(1,nrow(tr)-10):nrow(tr)))
print(tr[idx, c("iteration","log_raw_loss","shape","relative_gradient","log_density_norm","damping","rejections")], row.names=FALSE)
cat("initial theta:", paste(signif(fit$initial,6),collapse=" "), "\n")
cat("final theta:", paste(signif(fit$theta,6),collapse=" "), "\n")
cat("fit status:",fit$status,"iter",fit$iterations,"relgrad",fit$relative_gradient,"shape",fit$shape,"\n")
cat("input names:",paste(names(p$inputs),collapse=","),"\n")
inp <- p$inputs[[p$time]]
cat("time:",p$time,"points:",paste(dim(inp$points),collapse="x"),"\n")
cat("target range:",paste(range(inp$log_targets),collapse=" "),"\n")
cat("next twist names:",paste(names(inp$next_twist),collapse=","),"\n")
cat("next twist mean:",paste(signif(inp$next_twist$mean,5),collapse=" "),"\n")
cat("next twist covariance diag:",paste(signif(diag(inp$next_twist$covariance),5),collapse=" "),"\n")
cat("next twist log floor:",inp$next_twist$log_floor,"\n")
