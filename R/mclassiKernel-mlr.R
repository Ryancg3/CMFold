#' MLR Implementation Kernel
#' makeRLearner.classif.mclassiKernel
#' @importFrom mlr makeRLearner
#' @importFrom mlr predictLearner
#' @importFrom mlr trainLearner
#' @export
makeRLearner.classif.mclassiKernel <- function() {
  makeRLearnerClassif(
    cl = "classif.mclassiKernel",
    package = "CMFold",
    par.set = makeParamSet(
      makeDiscreteLearnerParam(id = "metric", default = "Euclidean", value = list("Euclidean", "L2", "Manhattan", "City-Block", "L1", "taxi", "Minkowski", "Lp", "supremum1", "supremum2", "infimum1", "infimum2", "mean", "minimum", "globMax", "globMin", "shortEuclidean", "Frechet", "Hausdorff", NULL = NULL)),
      makeIntegerLearnerParam(id = "nderiv", default = 0L, lower = 0L),
      makeDiscreteLearnerParam(id = "kernel", default = "Ker.norm", value = list("Ker.norm", "Ker.cos", "Ker.epa", "Ker.tri", "Ker.quar", "Ker.unif", NULL = NULL)),
      makeIntegerLearnerParam(id = "h", default = 1),
      makeLogicalLearnerParam(id = "baseline", default = FALSE, tunable = FALSE),
      makeNumericVectorLearnerParam(id = "weight", default = NULL, special.vals = list(NULL = NULL)),
      makeDiscreteLearnerParam(id = "measure", default = NULL, value = list("xpos_max", "xpos_min", "ypos_max", "ypos_min", "MAD", "MAD_time", "MD_above", "MD_above_time", "MD_below", "MD_below_time", "AD", "AUC", "xpos_flips", "ypos_flips", "flips", "xpos_reversals", "ypos_reversals", "RT", "initiation_time", "idle_time", "hover_time", "hovers", NULL = NULL)),
      makeLogicalLearnerParam(id = "reset_timestamps", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "resample", default = TRUE, tunable = FALSE),
      makeNumericLearnerParam(id = "flip_threshold", default = 0, lower = 0),
      makeNumericLearnerParam(id = "hover_threshold", default = 2000, lower = 0),
      makeLogicalLearnerParam(id = "parallel", default = FALSE, tunable = FALSE),
      makeIntegerLearnerParam(id = "cl", default = NULL, lower = 1L, special.vals = list(NULL = NULL), tunable = FALSE),
      makeLogicalLearnerParam(id = "diag", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "upper", default = FALSE, tunable = FALSE)
    ),
    # par.vals=list(keep.data=FALSE), #why???????
    par.vals = list(),
    properties = c(
      "twoclass",
      "multiclass",
      "numerics",
      "factors",
      "prob"
    ),
    name = "Multivariate functional kernel",
    short.name = "mclassiKernel",
    callees = "mclassiKernel"
  )
}

#' @export
trainLearner.classif.mclassiKernel <- function(.learner, .task, .subset, timestamps = NULL, ...) {
  ftask <- getTaskDesc(.task)
  fdata <- getTaskData(.task, .subset)
  # if (length(ftask$class.levels) == 2L) {
  #   fdata <- getTaskData(.task, .subset, recode.target = "01")
  # } else {
  #   fdata <- getTaskData(.task, .subset)
  # }
  model <- mclassiKernel(classes = fdata[, ftask$target], fdata = fdata[, -which(names(fdata) == ftask$target)], timestamps = timestamps, ...)
  return(model)
}

#' @export
predictLearner.classif.mclassiKernel <- function(.learner, .model, .newdata, ...) {
  predict.mclassiKernel(
    object = .model$learner.model,
    newdata = .newdata,
    predict.type = .model$learner$predict.type
  )
}
