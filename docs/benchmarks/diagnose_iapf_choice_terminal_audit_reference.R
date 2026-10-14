# Independent CPU terminal audit. Replay original solver and keep parameters.
args <- commandArgs(TRUE)
root<-args[1];refroot<-args[2];out<-args[3]
previous<-file.path(dirname(out),"attempt01")
snapshot<-file.path(previous,"sources/diagnose_iapf_choice_calibration_reference.R")
lines<-readLines(snapshot)
end<-which(startsWith(lines,"# Stage 1,"))[1]-1L
stopifnot(is.finite(end))
# Load the exact reviewed function definitions, before its experiment entrypoint.
eval(parse(text=lines[seq_len(end)]),envir=.GlobalEnv)
budget<-240
fixtures<-readRDS(file.path(previous,"fixtures.rds"))
old<-read.csv(file.path(previous,"fits.csv"))
dir.create(file.path(out,"fits"))
for(f in fixtures)if(f$kind%in%c("correlated","saved")) {
  initial<-iapf_fit_initial(f$x,f$y,TRUE)$parameters
  for(solver in c("nlminb","lm")) {
    fit<-solve_fit(f$x,f$y,initial,initial,"profiled",solver)
    saveRDS(list(fixture=f$id,initial=initial,fit=fit),
            file.path(out,"fits",paste0(f$id,"-",solver,".rds")))
    before<-old[old$fixture==f$id & old$arm==paste0("profiled_",solver,"_qr"),]
    stopifnot(nrow(before)==1,identical(before$status,fit$status),
      abs(before$shape-fit$shape)<1e-12,abs(before$loss-fit$loss)<1e-12,
      abs(before$logdensity_change-fit$logdensity_change)<1e-10)
    engine<-make_eval(f$x,f$y,initial,"profiled")
    p<-fit$parameters;ev<-engine$calc(p)
    density<-exp(logdiag(f$x,p)-engine$fixed)
    mass<-mean(density^2)/engine$normalization
    relative_gradient<-max(abs(ev$gradient))/mass
    lambda<-engine$lambda(p)
    cos_angle<-sqrt(max(0,1-shape(p,f$x,f$y)))
    # Fixed positive rescaling of the Hessian at the final point changes no
    # eigenvalue sign. Central derivative is a diagnostic, not solver feedback.
    h<-1e-4
    H<-vapply(seq_along(p),function(j){
      pp<-p;pm<-p
      pp[j]<-pp[j]+h*engine$coords[j];pm[j]<-pm[j]-h*engine$coords[j]
      (engine$calc(pp)$gradient-engine$calc(pm)$gradient)/(2*h*mass)
    },numeric(length(p)))
    asym<-max(abs(H-t(H)))/max(1,max(abs(H)))
    eig<-eigen((H+t(H))/2,symmetric=TRUE,only.values=TRUE)$values
    append_row(data.frame(fixture=f$id,kind=f$kind,d=f$d,solver=solver,
      original_status=fit$status,shape=fit$shape,
      initial_shape=shape(initial,f$x,f$y),
      absolute_gradient=fit$gradient,relative_gradient=relative_gradient,
      final_density_norm_ratio=mass,minimum_hessian_eigenvalue=min(eig),
      maximum_hessian_eigenvalue=max(eig),hessian_asymmetry=asym,
      logdensity_change=fit$logdensity_change,cosine_to_target=cos_angle,
      parity_pass=TRUE),"stationarity.csv")
  }
  cat("AUDIT",f$id,"PASS\n")
}
fl<-read.csv(file.path(previous,"floors.csv"))
heur<-read.csv(file.path(previous,"floor-heuristics.csv"))
stopifnot(nrow(fl)==324,nrow(old)==266)
for(d in c(1,5,20,80))for(policy in c("mix1","mix2")) {
  r<-fl[fl$d==d & fl$regime=="healthy" & fl$offset==0 & fl$policy==policy,]
  delta<-1000^(-as.numeric(substr(policy,4,4)))
  stopifnot(abs(r$base_probability-delta)<1e-12)
}
# Gaussian closed-form comparator must agree with no-floor quadrature.
for(regime in c("healthy","underdispersed","miscentered"))for(offset in c(0,1,3)) {
  r<-fl[fl$d==1 & fl$regime==regime & fl$offset==offset & fl$policy=="zero",]
  h<-heur[heur$regime==regime & heur$offset==offset & heur$method=="unfloored_fit",]
  if(is.finite(h$second_moment))stopifnot(abs(r$second_moment/h$second_moment-1)<1e-8)
  else stopifnot(is.infinite(r$second_moment))
}
healthy<-fl[fl$d==1 & fl$regime=="healthy",]
stopifnot(all(healthy$second_moment>=1-1e-7),
          all(healthy$second_moment<=healthy$healthy_second_moment_upper+1e-7))
# Report comparisons per regime, preserving the never-tune-on-heuristics role.
comparisons<-list()
for(i in which(fl$d==1)) {
  f<-fl[i,]
  h<-heur[heur$regime==f$regime & heur$offset==f$offset &
            heur$method!="exact_conditional",]
  for(j in seq_len(nrow(h)))comparisons[[length(comparisons)+1L]]<-
    data.frame(regime=f$regime,offset=f$offset,policy=f$policy,
      heuristic=h$method[j],candidate_second_moment=f$second_moment,
      heuristic_second_moment=h$second_moment[j],
      loses=is.finite(h$second_moment[j]) &&
        f$second_moment>h$second_moment[j]*(1+1e-8)+1e-8)
}
write.csv(do.call(rbind,comparisons),file.path(out,"conditional-heuristics.csv"),row.names=FALSE)
writeLines(c("status=PASS","parity_replays=26","floor_algebra_and_gaussian_integrals=PASS",
  "role=explanatory_terminal_audit","scientific_replication=INCOMPLETE"),
  file.path(out,"terminal.txt"))
cat("TERMINAL AUDIT PASS\n")
