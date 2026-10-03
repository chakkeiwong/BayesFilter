# Focused checks of recorded output, with no rerun or candidate retuning.
campaign <- "docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01"
cases <- read.csv(file.path(campaign,"attempt03-solver/results/solver-controls.csv"))
records <- cases[!duplicated(cases$case),,drop=FALSE]
unique_inputs <- list();group <- integer(nrow(records))
for(j in seq_len(nrow(records))) {
  e <- readRDS(records$source[j]);input <- list(points=e$points,log_targets=e$log_targets)
  matching <- which(vapply(unique_inputs,function(previous)identical(input,previous),logical(1)))
  if(!length(matching)) {unique_inputs[[length(unique_inputs)+1L]] <- input;matching <- length(unique_inputs)}
  group[j] <- matching[1]
}
write.csv(data.frame(case=records$case,source=records$source,distinct_input=group),
  file.path(out,"distinct-saved-inputs.csv"),row.names=FALSE)
for(j in records$case) {
  result <- readRDS(file.path(campaign,"attempt03-solver/results",paste0("case",j,".rds")))
  for(method in c("bounded1","ridge_bounded1")) {
    fit <- result$fits[[method]];d <- (length(fit$coefficients)-1)/2
    q <- fit$coefficients[1+d+seq_len(d)]
    violation <- max(0,q-fit$diagnostics$q_upper)
    check(paste0("saved_case",j,"_",method,"_bound"),violation,1e-14)
  }
}
rows <- list()
run_paths <- c(sort(Sys.glob(file.path(campaign,"attempt*-learning/results/learning.rds"))),
  sort(Sys.glob(file.path(campaign,"attempt*-bridge/results/repaired-learning.rds"))))
for(path in run_paths) {
  run <- readRDS(path)
  if(inherits(run,"error") || !identical(run$hypothesis,"independent_R_ridge_bounded_weighted_log_extension"))next
  for(iteration in seq_along(run$fits))for(time in seq_along(run$fits[[iteration]])) {
    d <- run$fits[[iteration]][[time]]
    rows[[length(rows)+1L]] <- data.frame(source=path,iteration=iteration-1,time=time,
      lambda=d$lambda,active_constraints=d$active_constraints,kkt=d$kkt_residual,
      condition=d$hessian_condition,condition_target=d$ridge_condition_target,
      active_set_iterations=v(d,"active_set_iterations",0),
      original_kkt=v(d,"original_kkt_residual",d$kkt_residual))
  }
}
tab <- do.call(rbind,rows)
check("learning_KKT",max(tab$kkt),1e-6)
check("derived_Hessian_condition_cap",max(0,tab$condition/tab$condition_target-1),1e-10)
write.csv(tab,file.path(out,"all-learned-fit-controls.csv"),row.names=FALSE)
writeLines(unique(c(records$source,run_paths)),
  file.path(out,"input-paths.txt"))
cat("PASS",length(checks),"recorded-output checks;",length(unique_inputs),
  "distinct saved point/target inputs among",nrow(records),"records\n")
