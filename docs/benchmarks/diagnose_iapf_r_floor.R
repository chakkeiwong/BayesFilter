# CPU-only independent reference: instrument frozen proposals without changing RNG.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(args[7]=="floor_diagnostic",!dir.exists(output))
dir.create(output,recursive=TRUE)
base <- file.path(root,"docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01")
saved <- file.path(base,"attempt06-d80-diagnostic/results")
model_data <- readRDS(file.path(saved,"model.rds"))
model <- model_data$model; kalman <- model_data$kalman
old <- readRDS(file.path(saved,"replayed-result.rds"))
prefixes <- read.csv(file.path(saved,"prefixes.csv"))
write.csv(model$observations,file.path(output,"observations.csv"),row.names=FALSE)
original_proposal <- iapf_proposal; original_draw <- iapf_draw_proposal
all_rows <- curve_rows <- summary_rows <- list()
for (call in c(15L,19L,20L)) {
  input <- readRDS(file.path(saved,paste0("filter-input-",call,".rds")))
  fit_N <- old$counts[call-1L]
  for (arm in c("original","zero_at_93","zero_all")) {
    twists <- input$twists
    if (arm=="zero_at_93") twists[[93]]$log_floor <- -Inf
    if (arm=="zero_all") for (t in seq_along(twists)) twists[[t]]$log_floor <- -Inf
    time <- 0L; diagnostics <- selections <- vector("list",model$horizon)
    iapf_proposal <- function(means,covariance,twist) {
      result <- original_proposal(means,covariance,twist)
      time <<- time+1L
      overlap <- iapf_log_normal(means,twist$mean,covariance+twist$covariance)
      diagnostics[[time]] <<- data.frame(call=call,iteration=call-1L,arm=arm,
        time=time,N=input$N,log_floor=twist$log_floor,
        floor_p_min=min(result$floor_probability),floor_p_mean=mean(result$floor_probability),
        floor_p_max=max(result$floor_probability),
        overlap_log_min=min(overlap),overlap_log_mean=mean(overlap),overlap_log_max=max(overlap))
      if (arm=="original") {
        peak <- -.5*(model$dimension*log(2*pi)+
          as.numeric(determinant(twist$covariance,logarithm=TRUE)$modulus))
        for (power in c(2,3,4,8,16)) {
          floor <- peak-.5*qchisq(fit_N^(-power),df=model$dimension,lower.tail=FALSE)
          probability <- plogis(floor-overlap)
          curve_rows[[length(curve_rows)+1L]] <<- data.frame(call=call,time=time,
            N=input$N,fit_N=fit_N,power=power,log_floor=floor,mean_probability=mean(probability),
            max_probability=max(probability),expected_floor_draws=sum(probability))
        }
        if (time==93) saveRDS(list(means=means,covariance=covariance,twist=twist),
          file.path(output,paste0("proposal93-call",call,".rds")))
      }
      result
    }
    iapf_draw_proposal <- function(proposal) {
      seed <- .Random.seed
      original <- if (proposal$constant || all(proposal$floor_probability==0))
        rep(FALSE,nrow(proposal$means)) else runif(nrow(proposal$means)) < proposal$floor_probability
      assign(".Random.seed",seed,envir=.GlobalEnv)
      selections[[time]] <<- original
      diagnostics[[time]]$floor_draws <<- sum(original)
      original_draw(proposal)
    }
    assign(".Random.seed",input$rng,envir=.GlobalEnv)
    value <- iapf_apf(model,twists,input$N,input$kappa,TRUE)
    parity <- NA
    if (arm=="original") {
      previous <- prefixes[prefixes$iteration==call-1L,]
      stopifnot(identical(value$log_likelihood,old$history[call]),
        max(abs(value$prefix-kalman$prefix-previous$log_prefix_error))<1e-8,
        max(abs(value$ess-previous$ess)/pmax(1,value$ess))<1e-10)
      parity <- TRUE
    }
    for (t in seq_len(model$horizon)) {
      x <- value$clouds[[t]]
      observation <- model$log_observation(x,model$observations[t,],t)
      future <- if(t<model$horizon) iapf_log_integral(model$transition_mean(x,t+1),
        model$transition_covariance,twists[[t+1]]) else rep(0,nrow(x))
      guide <- iapf_log_twist(x,twists[[t]])
      diagnostics[[t]]$ess <- value$ess[t]
      diagnostics[[t]]$increment_ess <- iapf_ess(observation+future-guide)
      diagnostics[[t]]$log_prefix_error <- value$prefix[t]-kalman$prefix[t]
      diagnostics[[t]]$floor_guide_dominant_fraction <- mean(twists[[t]]$log_floor >
        iapf_log_normal(x,twists[[t]]$mean,twists[[t]]$covariance))
      diagnostics[[t]]$floor_weight_mass <- if(any(selections[[t]]))
        exp(iapf_logsum(value$log_weights[[t]][selections[[t]]])-iapf_logsum(value$log_weights[[t]])) else 0
    }
    rows <- do.call(rbind,diagnostics)
    all_rows[[length(all_rows)+1L]] <- rows
    summary_rows[[length(summary_rows)+1L]] <- data.frame(call=call,arm=arm,
      N=input$N,log_error=value$log_likelihood-kalman$log_likelihood,
      min_ess=min(value$ess),ess93=value$ess[93],
      error_increment93=rows$log_prefix_error[93]-rows$log_prefix_error[92],
      floor_probability93=rows$floor_p_mean[93],floor_draws93=rows$floor_draws[93],
      baseline_parity=parity)
    cat("DONE",call,arm,"log_error",value$log_likelihood-kalman$log_likelihood,"\n")
    write.csv(do.call(rbind,summary_rows),file.path(output,"summary.csv"),row.names=FALSE)
    write.csv(do.call(rbind,all_rows),file.path(output,"proposal.csv"),row.names=FALSE)
    write.csv(do.call(rbind,curve_rows),file.path(output,"floor-curve.csv"),row.names=FALSE)
  }
}
stopifnot(nrow(do.call(rbind,all_rows))==900L,nrow(do.call(rbind,curve_rows))==1500L)
