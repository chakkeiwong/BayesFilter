# Independent public-source diagnostic. No top-level author experiment runs.
args <- commandArgs(trailingOnly = TRUE)
source_path <- args[1]
output_path <- args[2]
expressions <- parse(source_path)
required <- c("g", "mu_aux", "g_aux", "f_aux", "ESS", "Num",
              "psi_tilda", "psi_t", "APF", "Psi")
loaded <- character()
for (node in expressions) {
  if (is.call(node) && identical(node[[1]], as.name("<-")) &&
      is.symbol(node[[2]]) && as.character(node[[2]]) %in% required &&
      is.call(node[[3]]) && identical(node[[3]][[1]], as.name("function"))) {
    eval(node, envir = .GlobalEnv)
    loaded <- c(loaded, as.character(node[[2]]))
  }
}
stopifnot(setequal(loaded, required))

# Dependency substitutions only: Gaussian density and sampling using base R.
# The sampler uses Cholesky rather than MASS's eigenbasis, so RNG paths differ.
dmvnorm <- function(x, mean = rep(0, ncol(as.matrix(x))), sigma = diag(length(mean))) {
  x <- matrix(x, ncol = length(mean))
  residual <- sweep(x, 2, mean)
  factor <- chol(sigma)
  standardized <- t(backsolve(factor, t(residual), transpose = TRUE))
  exp(-0.5 * (length(mean) * log(2*pi) + rowSums(standardized^2)) -
        sum(log(diag(factor))))
}
capture_draws <- FALSE
last_mu <- last_covariance <- NULL
mvrnorm <- function(n, mu, Sigma) {
  last_mu <<- mu
  last_covariance <<- Sigma
  if (capture_draws) return(if (n == 1) mu else matrix(rep(mu, each=n), nrow=n))
  samples <- matrix(rnorm(n * length(mu)), nrow=n) %*% chol(Sigma)
  samples <- sweep(samples, 2, mu, "+")
  if (n == 1) as.vector(samples) else samples
}
optimizer_records <- list()
optimizer_control <- NULL
optim <- function(...) {
  arguments <- list(...)
  if (!is.null(optimizer_control)) arguments$control <- optimizer_control
  result <- do.call(stats::optim, arguments)
  optimizer_records[[length(optimizer_records)+1]] <<- result
  cat("OPTIM:", result$convergence, result$message, "\n")
  result
}
rows <- list()
emit <- function(case, field, value) {
  value <- as.numeric(value)
  stopifnot(all(is.finite(value)))
  rows[[length(rows)+1]] <<- data.frame(case=case, field=field,
      index=seq_along(value), value=value)
}
flat <- function(value) as.vector(t(as.matrix(value)))
find_node <- function(node, predicate) {
  if (predicate(node)) return(node)
  if (is.call(node) && identical(node[[1]], as.name("function"))) return(NULL)
  if (is.call(node) || is.expression(node) || is.pairlist(node)) {
    children <- as.list(node)
    for (i in seq_along(children)) {
      if (identical(children[[i]], quote(expr=))) next
      found <- find_node(children[[i]], predicate)
      if (!is.null(found)) return(found)
    }
  }
  NULL
}

d <- 2
Time <- 3
N <- 49
grid <- as.matrix(expand.grid(seq(-2, 2, length.out=7), seq(-1.8, 2.2, length.out=7)))
X <- array(0, c(Time, N, d))
for (t in 1:Time) X[t,,] <- grid
obs <- rbind(c(.3,-.2), c(-.4,.25), c(.2,.5))
A <- matrix(0, d, d)
fit <- Psi(1, obs, X)
emit("fit", "points", flat(grid))
emit("fit", "observations", flat(obs))
emit("fit", "parameters", flat(fit))
for (i in seq_along(optimizer_records)) {
  emit("fit", paste0("optimizer", i), c(optimizer_records[[i]]$convergence,
       optimizer_records[[i]]$counts, optimizer_records[[i]]$value))
}
optimizer_records <- list()
optimizer_control <- list(factr=1)
tight_fit <- Psi(1, obs, X)
emit("fit", "tight_parameters", flat(tight_fit))
for (i in seq_along(optimizer_records)) {
  emit("fit", paste0("tight_optimizer", i), c(optimizer_records[[i]]$convergence,
       optimizer_records[[i]]$counts, optimizer_records[[i]]$value))
}
optimizer_control <- NULL

fn_node <- find_node(body(Psi), function(node) is.call(node) &&
    identical(node[[1]], as.name("<-")) && identical(node[[2]], as.name("fn")))
stopifnot(!is.null(fn_node))
source_fn <- eval(fn_node[[3]])
t <- 1
target <- dmvnorm(grid, c(.3,-.2), diag(c(.7,1.4)))
psi <- matrix(rep(target, each=Time), nrow=Time)
emit("objective", "targets", target)
for (j in 1:3) {
  parameters <- list(c(-.2,.15,log(1.15),log(.7)),
                     c(.3,-.2,log(sqrt(.7)),log(sqrt(1.4))),
                     c(.8,-.6,log(.8),log(1.3)))[[j]]
  objective <- function(p) source_fn(c(p[1:d], exp(2*p[(d+1):(2*d)])), X, psi)
  step <- 1e-5
  gradient <- sapply(seq_along(parameters), function(k) {
    delta <- rep(0, length(parameters)); delta[k] <- step
    (objective(parameters+delta)-objective(parameters-delta))/(2*step)
  })
  emit(paste0("objective", j), "parameters", parameters)
  emit(paste0("objective", j), "loss", objective(parameters))
  emit(paste0("objective", j), "gradient", gradient)
}

A <- matrix(c(.42,.1764,.1764,.42), 2, 2)
psi_pa <- cbind(rbind(c(.3,-.2),c(-.4,.25),c(.2,.5)),
                rbind(c(.7,1.4),c(1.2,.6),c(.9,1.1)))
x <- c(.4,-.6)
emit("components", "A", flat(A))
emit("components", "psi", flat(psi_pa))
emit("components", "x", x)
capture_draws <- TRUE
invisible(mu_aux(psi_pa,1))
emit("initial", "mean", last_mu)
emit("initial", "covariance", flat(last_covariance))
invisible(f_aux(x,psi_pa,2))
emit("transition", "mean", last_mu)
emit("transition", "covariance", flat(last_covariance))
capture_draws <- FALSE
for (t in 1:Time) {
  emit(paste0("weight",t), "log_psi", log(psi_t(x,psi_pa,t)))
  emit(paste0("weight",t), "log_future", log(psi_tilda(x,psi_pa,t)))
  emit(paste0("weight",t), "log_weight", log(g_aux(obs[t,],x,t,psi_pa)))
}

fit_condition_node <- find_node(expressions, function(node) is.call(node) &&
    identical(node[[1]],as.name("if")) && grepl("l <= k",paste(deparse(node[[2]]),collapse=" "),fixed=TRUE))
double_condition_node <- find_node(expressions, function(node) is.call(node) &&
    identical(node[[1]],as.name("if")) && grepl("is.unsorted",paste(deparse(node[[2]]),collapse=" "),fixed=TRUE))
stopifnot(!is.null(fit_condition_node), !is.null(double_condition_node))
k <- 2; tau <- .1
histories <- list(partial=c(0,0), first_complete=c(0,0,0),
                 next_complete=c(0,0,0,0), oscillating=c(1,0,.5),
                 shifted=c(-1001,-1000,-999,-999))
for (name in names(histories)) {
  Z <- histories[[name]]; l <- length(Z); N <- rep(8,l)
  fit_again <- eval(fit_condition_node[[2]])
  double <- fit_again && eval(double_condition_node[[2]])
  emit(name,"history",Z)
  emit(name,"controller",c(Num(Z,l,k), as.numeric(fit_again), if(double) 16 else 8))
}

d <- 1; Time <- 3; N <- 32; kappa <- 1; Z <- 0
A <- matrix(0,1,1); obs <- matrix(c(.3,-.5,.8),ncol=1)
psi_pa <- cbind(obs,1)
set.seed(91801)
output <- APF(psi_pa,1,N)
emit("full_filter", "log_likelihood", output[[4]][1])
emit("full_filter", "exact", sum(dnorm(obs,0,sqrt(2),log=TRUE)))
emit("full_filter", "observations", obs)

# Paper-conformance probes: execute the same parsed source functions. Prescribed
# noise and ancestors test the finite Algorithm-5 arithmetic, not RNG parity.
d <- 1; Time <- 3; N <- 3; Z <- 0
A <- matrix(.42, 1, 1)
obs <- matrix(c(.1, .8, -.2), ncol=1)
psi_pa <- cbind(c(.3, -.4, .2), c(.7, 1.2, .9))
emit("paper_model", "observations", obs)
emit("paper_model", "twists", flat(psi_pa))
emit("paper_model", "transition", A)
path <- c(.2, -.5, .7)
emit("paper_path", "states", path)
capture_draws <- TRUE
for (step in seq_len(Time)) {
  if (step == 1) invisible(mu_aux(psi_pa,1)) else
    invisible(f_aux(path[step-1],psi_pa,step))
  emit("paper_path", paste0("proposal",step), c(last_mu,last_covariance))
  emit("paper_path", paste0("weight",step), g_aux(obs[step,],path[step],step,psi_pa))
}
capture_draws <- FALSE
noise <- c(-1.2, .3, 1.1, -.8, .5, 1.4, .2, -1., .7)
emit("paper_model", "noise", noise)
saved_mvrnorm <- mvrnorm
saved_sample <- base::sample
mvrnorm <- function(n, mu, Sigma) {
  count <- n * length(mu)
  selected <- noise[noise_cursor + seq_len(count)]
  stopifnot(length(selected) == count, all(is.finite(selected)))
  noise_cursor <<- noise_cursor + count
  values <- sweep(matrix(selected, nrow=n) %*% chol(Sigma), 2, mu, "+")
  if (n == 1) as.vector(values) else values
}
sample <- function(x, size, replace, prob) {
  stopifnot(replace, size == 3)
  sample_count <<- sample_count + 1
  emit(probe_case, paste0("ancestor_probabilities", sample_count), prob)
  selected <- list(c(3,1,2), c(2,3,1))[[sample_count]]
  emit(probe_case, paste0("ancestors", sample_count), selected)
  x[selected]
}
for (resample in c(FALSE, TRUE)) {
  probe_case <- if(resample) "paper_resample" else "paper_retain"
  kappa <- if(resample) 1 else 0
  noise_cursor <- sample_count <- 0
  result <- APF(psi_pa, 1, N)
  emit(probe_case, "clouds", flat(result[[2]][,,1]))
  emit(probe_case, "log_weights", flat(result[[3]]))
  emit(probe_case, "log_likelihood", result[[4]][1])
  emit(probe_case, "sample_calls", sample_count)
  emit(probe_case, "noise_consumed", noise_cursor)
}
mvrnorm <- saved_mvrnorm
sample <- saved_sample

# A prescribed nonzero floor represents the example family in equation (16).
# The public interface has no floor argument; report the discrepancy, do not
# silently inject a floor or modify its proposal/normalizers.
emit("paper_floor", "x", c(.4, -1.2, 4.))
emit("paper_floor", "positive_floor", .01)
emit("paper_floor", "twist", sapply(c(.4, -1.2, 4.), function(x) psi_t(x,psi_pa,1)))

# Observe actual Algorithm-3 targets passed from Psi to its nested objective.
# A deterministic optimizer substitute supplies the next twist and captures
# the target; optimization accuracy is tested separately above.
saved_optim <- optim
target_cloud <- c(-1.1, -.2, .4, 1.3)
Time <- 3; N <- 4
X <- array(rep(target_cloud, each=Time), c(Time, N, 1))
optim <- function(par, fn, X, psi, ...) {
  fit_time <- get("t", envir=parent.frame())
  emit("paper_backward", paste0("target", fit_time), psi[fit_time,])
  list(par=psi_pa[fit_time,])
}
invisible(Psi(1, obs, X))
emit("paper_backward", "cloud", target_cloud)
optim <- saved_optim

# Algorithm 4: execute the source while loop and final evaluation, replacing
# numerical APF/Psi only to supply controlled histories and record their calls.
while_indices <- which(vapply(as.list(expressions), function(node)
  is.call(node) && identical(node[[1]], as.name("while")), logical(1)))
stopifnot(length(while_indices) == 1)
loop_index <- while_indices[1]
final_node <- expressions[[loop_index+1]]
stopifnot(is.call(final_node), identical(final_node[[1]], as.name("<-")),
          identical(final_node[[2]], as.name("output")),
          identical(final_node[[3]][[1]], as.name("APF")))
saved_APF <- APF; saved_Psi <- Psi
APF <- function(psi_pa, l, N) {
  stopifnot(l <= length(controller_history), length(controller_events) < 20)
  controller_events[[length(controller_events)+1]] <<- c(l, N[l], psi_pa, final_phase)
  values <- Z
  values[l] <- controller_history[l] + if(final_phase) .123 else 0
  list(obs, X, matrix(0, 1, 1), values)
}
Psi <- function(l, obs, X) {
  fitted_iterations <<- c(fitted_iterations, l)
  l
}
controller_cases <- list(stable=c(0,0,0,0,0,0), oscillating=c(0,1,0,1,1,1,1,1))
for (name in names(controller_cases)) {
  controller_history <- controller_cases[[name]]
  k <- 2; tau <- .1; l <- 1; N <- 8; Z <- controller_history[1]
  index <- TRUE; psi_pa <- 0; final_phase <- FALSE
  controller_events <- list(); fitted_iterations <- integer()
  eval(expressions[[loop_index]])
  final_phase <- TRUE
  eval(final_node)
  emit(paste0("paper_outer_",name), "events", flat(do.call(rbind,controller_events)))
  emit(paste0("paper_outer_",name), "fits", fitted_iterations)
  emit(paste0("paper_outer_",name), "history", controller_history)
  emit(paste0("paper_outer_",name), "stop_index_zero_based", l-1)
  emit(paste0("paper_outer_",name), "final_value", output[[4]][l])
}
APF <- saved_APF; Psi <- saved_Psi

# The general mathematical filter also has a one-observation boundary. Catch
# any source error as a domain finding without discarding other evidence.
d <- 1; Time <- 1; N <- 3; Z <- 0; kappa <- 1
A <- matrix(.42, 1, 1); obs <- matrix(.3, ncol=1); psi_pa <- matrix(c(.3,1),nrow=1)
single <- tryCatch(APF(psi_pa,1,N), error=function(e) {
  cat("SINGLE_TIME_ERROR:", conditionMessage(e), "\n"); NULL
})
emit("paper_single_time", "executed", !is.null(single))
if (!is.null(single)) emit("paper_single_time", "log_likelihood", single[[4]][1])
write.csv(do.call(rbind, rows),output_path,row.names=FALSE)
cat("Loaded source functions:", paste(loaded,collapse=","), "\n")
cat("R version:", R.version.string, "\n")
