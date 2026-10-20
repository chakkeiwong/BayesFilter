# Independent CPU-only reference experiment. No production or superiority claim.
args <- commandArgs(TRUE); phase <- args[1]; out <- args[2]; dep <- args[3]
source(file.path(out,"sources/reference_iapf_paper.R"))
source(file.path(out,"sources/reference_iapf_scale_aware_fit.R"))
write_rows <- function(x,name) write.csv(x,file.path(out,name),row.names=FALSE,na="")
append_row <- function(x,name) {
  path <- file.path(out,name); exists <- file.exists(path)
  write.table(x,path,sep=",",row.names=FALSE,col.names=!exists,append=exists,na="")
}
phase_status <- function(status,detail) {
  write_rows(data.frame(phase=phase,status=status,detail=detail),"phase_status.csv")
  cat(phase,status,detail,"\n")
}
new_model <- function(seed) {
  data <- iapf_paper_data(d=5,horizon=100,seed=seed)
  list(data=data,model=iapf_linear_model(data$observations,data$A))
}
fit_row <- function(t,fit,arm,sweep) {
  val <- function(k) if(is.null(fit[[k]])) NA else fit[[k]]
  append_row(data.frame(arm=arm,sweep=sweep,time=t,status=fit$status,
    iterations=val("iterations"),relative_gradient=val("relative_gradient"),
    shape=val("shape"),log_raw_loss=val("log_raw_loss"),
    log_density_change=val("log_density_change"),log_absolute_gradient=val("log_absolute_gradient"),
    eigen_min=val("eigen_min"),eigen_max=val("eigen_max"),
    original_loss_descent=val("original_loss_descent"),retried=!is.null(fit$first_attempt)),"fits.csv")
}

if(phase=="controls") {
  results <- list(); cases <- list(); index <- 0
  for(d in c(1,5,20,80)) for(amplitude in c(.05,.20)) {
    index <- index+1; set.seed(9261000+index)
    truth <- c(seq(-.2,.2,length.out=d),seq(-.3,.3,length.out=d))
    # Purpose-built, well-conditioned recovery inputs, not filtering data.
    x <- sweep(matrix(rnorm(max(1000,20*d)*d),ncol=d)*(if(d==80).35 else .5),
      2,exp(truth[d+seq_len(d)]/2),"*")
    x <- sweep(x,2,truth[seq_len(d)],"+"); ly <- sa_log_density(x,truth)+2
    initial <- truth+amplitude*rnorm(2*d)
    fit <- sa_fit(x,ly,initial,curvature=d<=20)
    error <- max(abs(fit$theta-truth))
    results[[index]] <- data.frame(d=d,amplitude=amplitude,status=fit$status,
      iterations=fit$iterations,parameter_error=error,relative_gradient=fit$relative_gradient,
      descent=fit$original_loss_descent,pass=fit$status=="stationary" && error<=1e-4 &&
        fit$relative_gradient<=1e-6 && fit$original_loss_descent)
    cases[[index]] <- list(x=x,ly=ly,truth=truth,initial=initial,fit=fit)
  }
  write_rows(do.call(rbind,results),"recovery.csv"); saveRDS(cases,file.path(out,"controls.rds"))
  case <- cases[[3]]; prob <- sa_problem(case$x,case$ly,case$initial)
  z <- rep(0,length(case$initial)); ev <- prob$evaluate(z); h <- .Machine$double.eps^(1/3)
  fd <- sapply(seq_along(z),function(j) {
    dz <- z; dz[j] <- h
    (prob$evaluate(z+dz,ev$scale)$loss-prob$evaluate(z-dz,ev$scale)$loss)/(2*h)
  })
  fd_error <- max(abs(fd-ev$gradient))/max(1e-12,max(abs(ev$gradient)))
  shifted <- sa_fit(case$x,case$ly,case$initial,scale_offset=10)
  scale_error <- max(abs(shifted$theta-case$fit$theta))
  # Independent direct joint residual after inserting the profiled lambda.
  direct_p <- exp(sa_log_density(case$x,case$initial)); direct_y <- exp(case$ly)
  direct_lambda <- sum(direct_p*direct_y)/sum(direct_y^2)
  direct_loss <- mean((direct_p-direct_lambda*direct_y)^2)
  objective_error <- abs(exp(ev$log_raw_loss)-direct_loss)/direct_loss
  data <- iapf_paper_data(d=5,horizon=12,seed=9261099)
  model <- iapf_linear_model(data$observations,data$A); oracle <- iapf_kalman(model)
  set.seed(9261100); exact <- iapf_apf(model,iapf_exact_twists(model),32)
  kalman_error <- abs(exact$log_likelihood-oracle$log_likelihood)
  floor_rows <- list()
  for(power in c(2,1)) {
    delta <- 1000^(-power); twist <- sa_twist(c(0,0),1000,power,matrix(1))
    prop <- iapf_proposal(matrix(0),matrix(1),twist)
    logmix <- function(x,variance) iapf_logadd(log1p(-delta)+dnorm(x,0,sqrt(variance),log=TRUE),
      log(delta)+dnorm(x,0,1,log=TRUE))
    # Exact healthy target and correctly matched proposal, parent at peak.
    healthy <- integrate(function(x) exp(2*dnorm(x,0,sqrt(.5),log=TRUE)-logmix(x,.5)),
      -12,12,rel.tol=1e-10)$value
    stressed <- integrate(function(x) exp(2*dnorm(x,0,sqrt(.5),log=TRUE)-logmix(x,.02/1.02)),
      -12,12,rel.tol=1e-8)$value
    floor_rows[[length(floor_rows)+1]] <- data.frame(power=power,delta=delta,
      actual_peak_probability=prop$floor_probability,healthy_second_moment=healthy,
      healthy_bound=1/(1-delta),underdispersed_second_moment=stressed,
      pass=abs(prop$floor_probability-delta)<1e-12 && healthy<=1/(1-delta)+1e-10 && is.finite(stressed))
  }
  floor_table <- do.call(rbind,floor_rows); write_rows(floor_table,"floor_controls.csv")
  checks <- data.frame(check=c("gradient_fd","fixed_rescale_invariance","profiled_joint_objective","exact_twist_kalman"),
    error=c(fd_error,scale_error,objective_error,kalman_error),tolerance=c(1e-5,1e-7,1e-12,1e-9))
  checks$pass <- checks$error<=checks$tolerance; write_rows(checks,"checks.csv")
  recovery <- do.call(rbind,results)
  common_pass <- all(checks$pass) && all(floor_table$pass)
  all_pass <- common_pass && all(recovery$pass)
  d5_pass <- common_pass && all(recovery$pass[recovery$d<=5])
  write_rows(data.frame(scope=c("d5_diagnostic","d1_d5_d20_d80_recovery"),
    pass=c(d5_pass,all_pass)),"scope_qualification.csv")
  phase_status(if(all_pass)"PASS" else if(d5_pass)"PASS_D5_ONLY" else "FAIL",
    "Common controls and scope-specific recovery; broad recovery failures remain disqualifying outside d5")
  if(!d5_pass) quit(status=2)
} else if(phase=="repair") {
  cases <- readRDS(file.path(out,"sources/recovery_inputs.rds"))
  outcomes <- read.csv(file.path(dep,"backward_outcomes.csv"))
  failed <- outcomes[outcomes$status=="fit_failed",]
  stopifnot(nrow(failed)==4)
  diagnostics <- list(); nominees <- numeric()
  for(damping in c(1,1000,1000000)) {
    control_pass <- TRUE
    for(index in c(3,4)) {
      case <- cases[[index]]
      fit <- sa_fit(case$x,case$ly,case$initial,maxit=800,initial_damping=damping)
      parameter_error <- max(abs(fit$theta-case$truth))
      passed <- fit$status=="stationary" && parameter_error<=1e-4 &&
        fit$relative_gradient<=1e-6 && fit$original_loss_descent
      control_pass <- control_pass && passed
      append_row(data.frame(damping=damping,case=index,status=fit$status,
        parameter_error=parameter_error,relative_gradient=fit$relative_gradient,pass=passed),"repair_controls.csv")
    }
    candidate_pass <- control_pass
    for(i in seq_len(nrow(failed))) {
      old <- readRDS(file.path(dep,paste0(failed$arm[i],"-",failed$data_seed[i],".rds")))
      input <- old$inputs[[old$time]]
      initial <- old$fit$initial
      problem <- sa_problem(input$points,input$log_targets,initial)
      z <- rep(0,length(initial)); ev <- problem$evaluate(z); h <- .Machine$double.eps^(1/3)
      fd <- sapply(seq_along(z),function(j) {
        dz <- z; dz[j] <- h
        (problem$evaluate(z+dz,ev$scale)$loss-problem$evaluate(z-dz,ev$scale)$loss)/(2*h)
      })
      fd_error <- max(abs(fd-ev$gradient))/max(1e-12,max(abs(ev$gradient)))
      if(!is.finite(fd_error) || fd_error>1e-5) stop("Real-input derivative continuation veto")
      fit <- sa_fit(input$points,input$log_targets,initial,maxit=800,initial_damping=damping)
      passed <- fit$status=="stationary" && fit$relative_gradient<=1e-6 &&
        fit$original_loss_descent && is.finite(fit$eigen_min)
      candidate_pass <- candidate_pass && passed
      append_row(data.frame(damping=damping,arm=failed$arm[i],seed=failed$data_seed[i],
        status=fit$status,iterations=fit$iterations,relative_gradient=fit$relative_gradient,
        log_density_change=fit$log_density_change,shape=fit$shape,fd_error=fd_error,
        eigen_min=fit$eigen_min,original_loss_descent=fit$original_loss_descent,pass=passed),"repair_fits.csv")
      diagnostics[[paste(damping,i,sep="_")]] <- fit
    }
    if(candidate_pass) nominees <- c(nominees,damping)
    cat("damping",damping,"candidate_pass",candidate_pass,"\n")
  }
  saveRDS(diagnostics,file.path(out,"repair_trajectories.rds"))
  write_rows(data.frame(selected_damping=if(length(nominees))min(nominees) else NA,
    nominee_count=length(nominees),numerical_checks_pass=TRUE),"repair_decision.csv")
  phase_status(if(length(nominees))"PASS_REPAIR" else "COMPLETE_NO_NOMINEE",
    "Scope controls and actual-input derivatives checked; all failed candidates remain recorded")
} else if(phase=="backward") {
  controls <- read.csv(file.path(dep,"phase_status.csv"))
  stopifnot(controls$status %in% c("PASS","PASS_D5_ONLY","PASS_REPAIR"))
  initial_damping <- if(controls$phase=="repair") read.csv(file.path(dep,"repair_decision.csv"))$selected_damping else 1e-3
  stopifnot(is.finite(initial_damping),initial_damping>0)
  write_rows(data.frame(initial_damping=initial_damping),"configuration.csv")
  outcomes <- list()
  for(seed in c(9262001,9262002)) {
    bundle <- new_model(seed); model <- bundle$model
    set.seed(9264000+seed-9262000)
    baseline <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
    saveRDS(list(bundle=bundle,baseline=baseline),file.path(out,paste0("data-",seed,".rds")))
    for(power in c(2,1)) {
      arm <- paste0("min_mass_Nminus",power)
      snapshot <- file.path(out,paste0(arm,"-",seed,".rds"))
      started <- proc.time()[3]
      result <- tryCatch(sa_backward(model,baseline$clouds,1000,power,snapshot,
        function(t,fit)fit_row(t,fit,arm,seed),initial_damping=initial_damping),iapf_fit_error=function(e)e)
      success <- !inherits(result,"iapf_fit_error")
      outcomes[[length(outcomes)+1]] <- data.frame(arm=arm,power=power,data_seed=seed,
        status=if(success)"complete" else "fit_failed",time=if(success)NA else result$backward_time,
        elapsed_seconds=proc.time()[3]-started,detail=if(success)"100 actual backward fits" else conditionMessage(result))
      write_rows(do.call(rbind,outcomes),"backward_outcomes.csv")
      cat(arm,seed,if(success)"complete" else conditionMessage(result),"\n")
    }
  }
  phase_status("COMPLETE","Candidate failures are recorded; only arms passing both datasets may enter pilot")
} else if(phase=="pilot") {
  backward <- read.csv(file.path(dep,"backward_outcomes.csv"))
  config <- file.path(dep,"configuration.csv")
  initial_damping <- if(file.exists(config)) read.csv(config)$initial_damping else 1e-3
  eligible <- unique(backward$power)[vapply(unique(backward$power),function(p)
    sum(backward$power==p)==2 && all(backward$status[backward$power==p]=="complete"),TRUE)]
  bundle <- new_model(9263001); model <- bundle$model; oracle <- iapf_kalman(model)
  saveRDS(list(bundle=bundle,oracle=oracle),file.path(out,"untouched_data.rds"))
  pilots <- list(); outcomes <- list()
  for(power in eligible) {
    arm <- paste0("min_mass_Nminus",power); sweep <- 0L
    fit_callback <- function(model,clouds,N,maxit,floor_tail_power) {
      sweep <<- sweep+1L
      sa_backward(model,clouds,N,power,file.path(out,paste0(arm,"-sweep",sweep,".rds")),
        function(t,fit)fit_row(t,fit,arm,sweep),initial_damping=initial_damping)
    }
    set.seed(9264000); started <- proc.time()[3]
    result <- tryCatch(iapf_iterate(model,N0=1000,k=5,tau=.5,kappa=.5,max_iterations=20,
      max_particles=16000,fit_maxit=400,.fit=fit_callback,doubling_mode="after_k",
      stopping_window=6,cv_sd_mode="sample"),iapf_fit_error=function(e)e)
    elapsed <- proc.time()[3]-started; failed <- inherits(result,"iapf_fit_error")
    status <- if(failed)"fit_failed" else result$status
    saveRDS(result,file.path(out,paste0(arm,"-pilot.rds")))
    outcomes[[length(outcomes)+1]] <- data.frame(arm=arm,power=power,status=status,
      seconds=elapsed,iteration=if(failed)result$iteration else if(status=="complete")result$stop_iteration else NA,
      N=if(status=="complete")result$final_particles else NA,
      final_error=if(status=="complete")result$final$log_likelihood-oracle$log_likelihood else NA,
      detail=if(failed)conditionMessage(result) else status)
    write_rows(do.call(rbind,outcomes),"pilot_outcomes.csv")
    if(status=="complete") pilots[[arm]] <- result
    cat(arm,status,"elapsed",elapsed,"\n")
  }
  if(!length(outcomes)) write_rows(data.frame(arm=character(),status=character()),"pilot_outcomes.csv")
  particle_counts <- unique(c(1000,vapply(pilots,function(x)x$final_particles,0)))
  large <- oracle$innovation>qchisq(.95,5)
  # Same random seed is only a reproducibility convention, not a claim of CRN pairing.
  for(N in particle_counts) {
    methods <- c("BPF","observation_only","fully_adapted",names(pilots)[vapply(pilots,function(p)p$final_particles==N,TRUE)])
    for(method in methods) for(draw in seq_len(8)) {
      set.seed(9265000+draw); started <- proc.time()[3]
      result <- if(method=="fully_adapted") iapf_fully_adapted(model,N) else {
        twists <- if(method=="BPF") replicate(100,iapf_constant_twist(),simplify=FALSE) else
          if(method=="observation_only") iapf_observation_twists(model) else pilots[[method]]$twists
        iapf_apf(model,twists,N)
      }
      seconds <- proc.time()[3]-started
      errors <- diff(c(0,result$prefix))-diff(c(0,oracle$prefix))
      stopifnot(all(is.finite(errors)),is.finite(result$log_likelihood))
      append_row(data.frame(method=method,N=N,draw=draw,seconds=seconds,
        total_error=result$log_likelihood-oracle$log_likelihood,
        ordinary_mse=mean(errors[!large]^2),large_mse=if(any(large))mean(errors[large]^2) else NA,
        n_large=sum(large),n_ordinary=sum(!large),
        floor_probability_max=if(is.null(result$floor_probability_max))NA else max(result$floor_probability_max)),"evaluation.csv")
      append_row(data.frame(method=method,N=N,draw=draw,time=seq_len(100),
        large_innovation=large,increment_error=errors),"conditional_errors.csv")
    }
  }
  phase_status("COMPLETE",paste(length(pilots),"complete adaptive pilots; diagnostic 8-draw comparator evaluation"))
} else stop("unknown phase")
