# Post-run statistical diagnostic; no fitting or runtime decision role.
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args)%in%c(5,7))
planned_cells <- if(length(args)==7)as.integer(args[6])else 44L
planned_replicates <- if(length(args)==7)as.integer(args[7])else 512L
fixture <- args[1]; out <- args[5]; dir.create(out,recursive=TRUE)
data <- do.call(rbind,lapply(args[2:4],function(p)read.csv(file.path(p,"records.csv"))))
stopifnot(nrow(data)==planned_cells*planned_replicates*3, all(is.finite(data$normalized_likelihood)),
  all(is.finite(data$log_likelihood)),!anyDuplicated(data[c("cell","arm","mode","replicate")]))
means <- list(); comparisons <- list(); heuristics <- list()
contrast_count <- planned_cells*2
set.seed(2026092318)
for(cell in unique(data$cell))for(arm in unique(data$arm)) {
  sub <- data[data$cell==cell & data$arm==arm,]
  for(mode in c("fp64","fp32","tf32")) {
    x <- sub[sub$mode==mode,]; z <- x$normalized_likelihood
    stopifnot(nrow(x)==planned_replicates,all(sort(x$replicate)==0:(planned_replicates-1)))
    square <- (z-1)^2
    means[[length(means)+1]] <- data.frame(cell=cell,arm=arm,mode=mode,n=length(z),
      mean=mean(z),mcse=sd(z)/sqrt(length(z)),relative_rmse=sqrt(mean(square)),
      max_likelihood=max(z),largest_squared_error_share=if(sum(square)>0)max(square)/sum(square)else 0,
      mean_log_error=mean(x$log_likelihood-x$kalman))
  }
  candidate <- sub[sub$mode=="tf32",]; candidate <- candidate[order(candidate$replicate),]
  for(mode in c("fp64","fp32")) {
    baseline <- sub[sub$mode==mode,]; baseline <- baseline[order(baseline$replicate),]
    stopifnot(identical(candidate$seed,baseline$seed),identical(candidate$replicate,baseline$replicate))
    delta <- candidate$normalized_likelihood-baseline$normalized_likelihood
    estimate <- mean(delta); se <- sd(delta)/sqrt(length(delta))
    half <- qt(1-.01/(2*contrast_count),length(delta)-1)*se
    square <- (delta-estimate)^2
    tail <- if(sum(square)>0)max(square)/sum(square)else 0
    boot <- replicate(1999,mean(sample(delta,replace=TRUE)))
    qi <- quantile(boot,c(.005,.995),names=FALSE)
    passed <- estimate-half > -.02 && estimate+half < .02 && tail<=.2
    comparisons[[length(comparisons)+1]] <- data.frame(cell=cell,arm=arm,comparator=mode,
      difference=estimate,mcse=se,lower=estimate-half,upper=estimate+half,margin=.02,
      largest_centered_squared_share=tail,tail_veto=tail>.2,
      bootstrap_pointwise99_lower=qi[1],bootstrap_pointwise99_upper=qi[2],
      equivalence=passed,confidence_family=.99,contrasts=contrast_count,n=length(delta))
  }
}
for(cell in unique(data$cell))for(mode in c("fp64","fp32","tf32")) {
  sub <- data[data$cell==cell & data$mode==mode,]
  candidate <- sub[sub$arm=="stress",]; candidate <- candidate[order(candidate$replicate),]
  for(arm in c("constant","observation","oracle")) {
    baseline <- sub[sub$arm==arm,]; baseline <- baseline[order(baseline$replicate),]
    delta <- (candidate$normalized_likelihood-1)^2-(baseline$normalized_likelihood-1)^2
    heuristics[[length(heuristics)+1]] <- data.frame(cell=cell,mode=mode,comparator=arm,
      paired_mse_difference=mean(delta),observed_quality_promotion_veto=mean(delta)>0,
      inference="descriptive_only_no_ranking",purpose="stress_fixture_not_trained_candidate")
  }
}
write.csv(do.call(rbind,means),file.path(out,"conditional-summary.csv"),row.names=FALSE)
write.csv(do.call(rbind,comparisons),file.path(out,"precision-comparisons.csv"),row.names=FALSE)
write.csv(do.call(rbind,heuristics),file.path(out,"heuristic-comparisons.csv"),row.names=FALSE)
write.csv(data,file.path(out,"records.csv"),row.names=FALSE)
writeLines(c(paste("99% simultaneous approximate Student intervals; Bonferroni",contrast_count,"contrasts."),
  "Bootstrap intervals are pointwise sensitivity diagnostics, not simultaneous evidence.",
  "Tail share above 0.2 vetoes equivalence. All claims conditional on fixed datasets and guides.",
  "Finite-label derivatives and strict path parity do not target the marginal score.",
  "CPU-only post-run R diagnostic; GPU intentionally hidden.",capture.output(sessionInfo())),
  file.path(out,"inference-method.txt"))
cat("records",nrow(data),"contrasts",length(comparisons),"\n")
