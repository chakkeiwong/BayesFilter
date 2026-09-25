# Known-target check of the frozen independent R initialized Eq15 fit.
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1)
source("docs/benchmarks/reference_iapf_paper.R")
source(file.path(args[1],"R-input.R"))
rows <- list();fits <- list()
for(f in fixtures) {
  d <- f$d; points <- matrix(unlist(f$points),1000,d,byrow=TRUE)
  target <- unlist(f$targets); center <- unlist(f$center); variance <- unlist(f$variance)
  initial <- iapf_fit_initial(points,target)
  fit <- tryCatch(iapf_fit_gaussian(points,target,N=1000,fit_mode="paper_eq15"),error=function(e)e)
  ok <- !inherits(fit,"error")
  row <- data.frame(d=d,seed=f$seed,regime=f$regime,success=ok,
    initialization=initial$method,
    QR_center_error=max(abs(initial$parameters[1:d]-center)),
    QR_variance_error=max(abs(exp(initial$parameters[d+1:d])-variance)),
    fitted_center_error=if(ok)max(abs(fit$twist$mean-center))else NA_real_,
    fitted_variance_error=if(ok)max(abs(diag(fit$twist$covariance)-variance))else NA_real_,
    shape_residual=if(ok)fit$diagnostics$relative_residual else NA_real_,
    evaluations=if(ok)fit$diagnostics$evaluations else NA_integer_,
    error=if(ok)""else conditionMessage(fit))
  rows[[length(rows)+1]] <- row;fits[[length(fits)+1]] <- fit
}
table <- do.call(rbind,rows)
write.table(table,file.path(args[1],"R-checks.tsv"),sep="\t",row.names=FALSE,quote=TRUE)
saveRDS(fits,file.path(args[1],"R-fits.rds"))
writeLines(capture.output(sessionInfo()),file.path(args[1],"R-session.txt"))
cat(sprintf("%d/%d R reference fits completed\n",sum(table$success),nrow(table)))
