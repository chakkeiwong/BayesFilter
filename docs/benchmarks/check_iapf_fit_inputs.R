# Independent base-R evaluation of the profiled density objective and its
# total analytic gradient. Diagnostic only; no optimizer or filter replacement.
args <- commandArgs(trailingOnly=TRUE)
source(args[1])
vec <- function(x) as.numeric(unlist(x))
mat <- function(x) do.call(rbind,lapply(x,vec))
error <- function(a,b) max(abs(a-b))/(1+max(abs(a),abs(b)))
rows <- list()
for (item in input) {
  z <- mat(item$z); logb <- vec(item$log_target); b <- exp(logb)
  n <- nrow(z); d <- ncol(z)
  for (state in names(item$states)) {
    expected <- item$states[[state]]; par <- vec(expected$parameters)
    center <- par[seq_len(d)]; lsd <- par[d+seq_len(d)]
    u <- sweep(sweep(z,2,center,'-'),2,exp(-lsd),'*')
    logp <- -sum(lsd)-rowSums(u*u)/2
    a <- max(logp); p <- exp(logp-a)
    lambda <- sum(p*b)/sum(b*b)
    residual <- p-lambda*b
    amplitude_squared <- exp(2*a)
    energy <- amplitude_squared*mean(p*p)
    loss <- amplitude_squared*mean(residual*residual)
    shape <- sum(residual*residual)/sum(p*p)
    score <- cbind(sweep(u,2,exp(-lsd),'*'),u*u-1)
    gradient <- as.vector(2*amplitude_squared*crossprod(score,p*residual)/n)
    shape_gradient <- as.vector(2*crossprod(score,p*residual-shape*p*p)/sum(p*p))
    rows[[length(rows)+1L]] <- data.frame(key=item$key,state=state,
      loss_error=error(loss,vec(expected$loss)),
      energy_error=error(energy,vec(expected$energy)),
      shape_error=error(shape,vec(expected$shape)),
      gradient_error=error(gradient,vec(expected$gradient)),
      shape_gradient_error=error(shape_gradient,vec(expected$shape_gradient)),
      identity_error=error(loss,energy*shape))
  }
}
answer <- do.call(rbind,rows)
write.csv(answer,args[2],row.names=FALSE)
if (nrow(answer)!=384L || any(!is.finite(as.matrix(answer[,3:8]))) ||
    max(as.matrix(answer[,3:8]))>1e-8) stop('Objective arithmetic mismatch')
cat('PASS:',nrow(answer),'independent objective/gradient rows\n')
