# Selected-case cap extension: diagnostic only, never replaces confirmation.
args<-commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3,args[1]=="cap_extension",Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
out<-args[2];original<-args[3];dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
source("docs/benchmarks/reference_iapf_constrained_diagnostic.R")
failures<-read.csv(file.path(original,"learning.csv"))
failures<-failures[failures$status!="complete",]
stopifnot(nrow(failures)==2,all(failures$status=="iteration_cap"),
          all(failures$method=="ridge_bounded1"),all(failures$dimension==80))
one<-function(i) {
  z<-failures[i,];directory<-file.path(original,paste0("d80-j",z$data_index,"-r",z$replication))
  baseline<-readRDS(file.path(directory,"ridge_bounded1-learning.rds"))
  data<-iapf_paper_data(80,100,seed=z$data_seed)
  model<-iapf_linear_model(data$observations,data$A);truth<-iapf_kalman(model)
  set.seed(z$learning_seed);start<-proc.time()[[3]]
  result<-iapf_constrained_diagnostic_run(model,max_iterations=24,max_particles=4000)
  seconds<-proc.time()[[3]]-start
  n<-length(baseline$history)
  error<-max(abs(result$history[seq_len(n)]-baseline$history))
  prefix_ok<-is.finite(error) && error<=1e-9 && identical(result$counts[seq_len(n)],baseline$counts)
  saveRDS(result,file.path(out,paste0("j",z$data_index,"-r",z$replication,".rds")))
  summary<-data.frame(dimension=80,data_index=z$data_index,replication=z$replication,
    data_seed=z$data_seed,learning_seed=z$learning_seed,status=result$status,
    prefix_equal=prefix_ok,prefix_max_error=error,iterations=length(result$history),
    final_particles=tail(result$counts,1),last_cv=iapf_likelihood_cv(tail(result$history,6),"sample"),
    diagnostic_terminal_log_error=if(result$status=="complete")result$final$log_likelihood-truth$log_likelihood else NA_real_,
    seconds=seconds,confirmation_replaced=FALSE)
  stopifnot(prefix_ok);summary
}
results<-parallel::mclapply(seq_len(nrow(failures)),one,mc.cores=2,mc.set.seed=FALSE)
stopifnot(all(vapply(results,is.data.frame,TRUE)))
write.csv(do.call(rbind,results),file.path(out,"cap-extension.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
print(do.call(rbind,results),row.names=FALSE)
