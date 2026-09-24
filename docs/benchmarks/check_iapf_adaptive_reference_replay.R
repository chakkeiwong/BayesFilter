# Independent reference check of the parameterized batch harness.
args <- commandArgs(trailingOnly=TRUE);actual <- args[1];expected <- args[2];out <- args[3]
options(digits=17)
a <- readRDS(file.path(actual,'inputs.rds'));b <- readRDS(file.path(expected,'inputs.rds'))
stopifnot(identical(a$data$observations,b$data$observations),identical(a$data$A,b$data$A),
  length(a$data$seed)==1L,length(b$data$seed)==1L,a$data$seed==b$data$seed)
checks <- list()
for(method in c('score_after_k','qr_after_k')) {
  a <- readRDS(file.path(actual,paste0('learner_',method,'_r01.rds')))
  b <- readRDS(file.path(expected,paste0('learner_',method,'_r01.rds')))
  stopifnot(identical(a$result$status,b$result$status),
    identical(a$result$counts,b$result$counts),identical(a$result$history,b$result$history),
    identical(a$result$final,b$result$final),identical(a$result$twists,b$result$twists),
    identical(a$result$fits,b$result$fits),identical(a$filter_calls,b$filter_calls),
    length(a$fit_history)==length(b$fit_history))
  for(i in seq_along(a$fit_history)) {
    a$fit_history[[i]]$cpu_seconds <- b$fit_history[[i]]$cpu_seconds <- NULL
    stopifnot(identical(a$fit_history[[i]],b$fit_history[[i]]))
  }
  checks[[method]] <- data.frame(method=method,exact_history=TRUE,exact_guides=TRUE,
    exact_final=TRUE,exact_fits=TRUE,exact_filter_diagnostics=TRUE)
}
for(method in c('bootstrap','fully_adapted','current_observation','full_oracle')) {
  a <- readRDS(file.path(actual,paste0('heuristic_',method,'_r01.rds')))
  b <- readRDS(file.path(expected,paste0('heuristic_',method,'_r01.rds')))
  stopifnot(identical(a,b))
}
write.csv(do.call(rbind,checks),out,row.names=FALSE)
cat('Exact replay passed for both learners and four heuristic controls\n')
