#' MLR Implementation KNN
#' makeRLearner.classif.mclassiKnn
#' @importFrom mlr makeRLearner
#' @importFrom mlr predictLearner
#' @importFrom mlr trainLearner
#' @export
makeRLearner.classif.mclassiKnn <- function() {
  makeRLearnerClassif(
    cl = "classif.mclassiKnn",
    package = "CMFold",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "knn", default = 1L, lower = 1L, upper = 99L),
      makeDiscreteLearnerParam(id = "metric", default = "Euclidean", value = list("Euclidean", "L2", "Manhattan", "City-Block", "L1", "taxi", "Minkowski", "Lp", "supremum1", "supremum2", "infimum1", "infimum2", "mean", "minimum", "globMax", "globMin", "shortEuclidean", "Frechet", "Hausdorff", "dtw")), # ,NULL=NULL)),
      makeIntegerLearnerParam(id = "nderiv", default = 0L, lower = 0L),
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
      makeLogicalLearnerParam(id = "upper", default = FALSE, tunable = FALSE),
      makeUntypedLearnerParam(id = "mdist", default = NULL),
      makeUntypedLearnerParam(id = "timestamps", default = NULL),
      makeDiscreteLearnerParam(id = "predict.type", default = "prob", value = list("response", "prob"))
    ),
    # par.vals=list(keep.data=FALSE),
    properties = c(
      "twoclass",
      "multiclass",
      "numerics",
      "factors",
      "prob"
    ),
    name = "Multivariate functional knn",
    short.name = "mclassiKnn",
    callees = "mclassiKnn"
  )
}

#' @export
trainLearner.classif.mclassiKnn <- function(.learner, .task, .subset, ...) {
  ftask <- getTaskDesc(.task)
  fdata <- getTaskData(.task, .subset)
  mclassiKnn(classes = fdata[, ftask$target], fdata = fdata[, -which(names(fdata) == ftask$target)], ...)
}

#' @export
predictLearner.classif.mclassiKnn <- function(.learner, .model, .newdata, ...) {
  if (!is.null(.model$learner.model$mdist)) {
    msubset <- .model$learner.model$mdist[!((1:ncol(.model$learner$par.vals$mdist)) %in% .model$subset), .model$subset]
  } else {
    msubset <- NULL
  }
  predict.mclassiKnn(
    object = .model$learner.model,
    newdata = .newdata,
    msubset = msubset,
    predict.type = .model$learner$par.vals$predict.type
  )
}

# .model=model
