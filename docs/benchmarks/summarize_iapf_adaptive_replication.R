# Post-run independent-reference statistics; no algorithm/default selection.
args <- commandArgs(trailingOnly=TRUE);input <- args[1];out <- args[2];stage <- as.integer(args[3])
options(digits=17)
records <- read.csv(input,stringsAsFactors=FALSE);B <- 2000L
methods <- c('score_after_k','qr_after_k','bootstrap','fully_adapted','current_observation','full_oracle')
summaries <- comparisons <- list()
for(d in c(5,10,20,40,80)) {
  by_method <- list()
  for(j in seq_along(methods)) {
    method <- methods[j];x <- records[records$d==d & records$method==method,]
    x <- x[order(x$replicate),];stopifnot(identical(as.integer(x$replicate),seq_len(stage)))
    complete <- all(x$status=='complete');n_complete <- sum(x$status=='complete')
    row <- data.frame(d=d,method=method,n=nrow(x),completed=n_complete,complete=complete,
      mean_particles=mean(x$particles),mean_resampling_count=if(complete)mean(x$resampling_count)else NA_real_,
      cpu_seconds=sum(x$cpu_seconds),ratio_mean=NA_real_,ratio_sd=NA_real_,relative_rmse=NA_real_,
      log_mse=NA_real_,mean_lower=NA_real_,mean_upper=NA_real_,sd_lower=NA_real_,sd_upper=NA_real_,
      rmse_lower=NA_real_,rmse_upper=NA_real_,ratio_min=NA_real_,ratio_q01=NA_real_,ratio_median=NA_real_,
      ratio_q99=NA_real_,ratio_max=NA_real_,maximum_loss_share=NA_real_,ratio_underflow_count=NA_integer_)
    if(complete) {
      v <- x$ratio;loss <- (v-1)^2
      stopifnot(all(is.finite(v)),all(is.finite(loss)))
      set.seed(974000+10*stage+100*d+j)
      bootstrap <- replicate(B,{sampled <- sample(v,length(v),replace=TRUE)
        c(mean(sampled),sd(sampled),sqrt(mean((sampled-1)^2)))})
      ci <- t(apply(bootstrap,1,quantile,probs=c(.025,.975),names=FALSE))
      row$ratio_mean <- mean(v);row$ratio_sd <- sd(v);row$relative_rmse <- sqrt(mean(loss))
      row$log_mse <- mean(x$log_error^2)
      row$mean_lower <- ci[1,1];row$mean_upper <- ci[1,2]
      row$sd_lower <- ci[2,1];row$sd_upper <- ci[2,2]
      row$rmse_lower <- ci[3,1];row$rmse_upper <- ci[3,2]
      row$ratio_min <- min(v);row$ratio_max <- max(v)
      row$ratio_q01 <- quantile(v,.01,names=FALSE);row$ratio_median <- median(v)
      row$ratio_q99 <- quantile(v,.99,names=FALSE)
      row$maximum_loss_share <- if(sum(loss)>0)max(loss)/sum(loss)else 0
      row$ratio_underflow_count <- sum(v==0)
    }
    summaries[[length(summaries)+1L]] <- row;by_method[[method]] <- x
  }
  for(j in 1:2)for(h in 3:5) {
    candidate <- by_method[[methods[j]]];control <- by_method[[methods[h]]]
    complete <- all(candidate$status=='complete') && all(control$status=='complete')
    difference <- lower <- upper <- NA_real_
    if(complete) {
      delta <- (candidate$ratio-1)^2-(control$ratio-1)^2
      set.seed(975000+10*stage+100*d+10*j+h)
      boots <- replicate(B,mean(sample(delta,length(delta),replace=TRUE)))
      difference <- mean(delta);ci <- quantile(boots,c(.025,.975),names=FALSE)
      lower <- ci[1];upper <- ci[2]
    }
    comparisons[[length(comparisons)+1L]] <- data.frame(d=d,method=methods[j],comparator=methods[h],
      complete=complete,mean_squared_error_difference=difference,lower=lower,upper=upper,
      observed_heuristic_loss=if(complete)difference>0 else NA,
      promotion_veto=!complete || difference>0,ranking_claim=FALSE)
  }
}
write.csv(do.call(rbind,summaries),file.path(out,'conditional-summary.csv'),row.names=FALSE)
write.csv(do.call(rbind,comparisons),file.path(out,'paired-comparisons.csv'),row.names=FALSE)
write.csv(data.frame(stage=stage,bootstrap_resamples=B,confidence=.95,interval='pointwise_empirical_percentile',
  scope='Monte_Carlo_conditional_on_one_data_set_per_dimension',simultaneous=FALSE,
  ranking_or_default_promotion=FALSE),file.path(out,'uncertainty-method.csv'),row.names=FALSE)
cat('Stage statistics complete:',stage,'labels per dimension; pointwise intervals only\n')
