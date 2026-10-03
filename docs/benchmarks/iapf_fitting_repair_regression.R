# Existing reference regression suite, with exact child command preserved.
command <- c("--vanilla",normalizePath("tests/reference_iapf_author_choices.R"),getwd(),getwd())
writeLines(c("/usr/bin/Rscript",command),file.path(out,"regression-command.txt"))
exit <- system2("/usr/bin/Rscript",command,stdout=file.path(out,"regression.log"),stderr=file.path(out,"regression.log"))
check("existing_author_choice_suite",exit,0)
