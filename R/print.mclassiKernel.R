#' print.mclassiKernel
print.mclassiKernel=function(x,...) {
  cat("\n")
  cat("\t mclassiKernel object \n")
  cat("\n")
  cat("data: \n")
  cat("",length(x$fdata),"dimensions","\n")
  cat("", length(levels(x$classes)), "classes:", levels(x$classes), "\n")
  cat("", ncol(x$fdata[[1]]), "observations of length", nrow(x$fdata[[1]]), "\n")
  cat("algorithm: \n")
  cat(" metric=", x$metric, "\n")
  cat(" measure=", x$measure, "\n")
  cat(" h=", x$h, "\n")
  cat(" nderiv=", x$nderiv, "\n")
  cat(" baseline=", x$baseline, "\n")
  cat(" weights=", object$weight, "\n")
  cat("\n")
}
