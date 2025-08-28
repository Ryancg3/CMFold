#' Predictions with an mclassiKnn object
#'
#' The present function yields predictions for new data with an mclassiKnn object .
#'
#' @param object mknn object defined with \code{\link[classiMultiFuncTest:mclassiKnn]{mclassiKnn}}
#' @param newdata  the functional covariates for prediction as a \code{list} of \eqn{m} objects in \code{matrix} form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points as the training data.
#' @param predict.type \code{string} to specify type of prediction. Either \code{"response"} or \code{"prob"}
#' @examples
#'
#' ## 3-dimensional functions
#' # classes
#' classes <- as.factor(c("Cat", "Dog", "Dog", "Cat"))
#'
#' # training_data
#' x <- replicate(4, rnorm(100, 0, 3))
#' y <- replicate(4, rnorm(100, 3, 1))
#' z <- replicate(4, rpois(100, 2))
#' training_data <- list(x, y, z)
#'
#' # Creating object for mknn prediction
#' object <- mclassiKnn(classes = classes, fdata = training_data, knn = 3, nderiv = 0, cl = NULL)
#'
#' # test_data
#' x <- replicate(2, rnorm(100, 0, 2.9))
#' y <- replicate(2, rnorm(100, 3, 1.5))
#' z <- replicate(2, rpois(100, 3))
#'
#' test_data <- list(x, y, z)
#'
#' # Predict
#'
#' predict.mclassiKnn(object = object, newdata = test_data, predict.type = "response")
#' @export

predict.mclassiKnn <- function(object, newdata = NULL, msubset = NULL, predict.type = "prob") {
  d <- unique(newdata[, "dim"])
  newdata <- t(newdata[, -which(names(newdata) %in% c("id", "dim"))])
  newdata <- plyr::alply(cbind(1:d), 1, function(i) cbind(data.frame(dim = rep(1:d, each = nrow(newdata) / d), newdata)[data.frame(dim = rep(1:d, each = nrow(newdata) / d), newdata)[, "dim"] == i, -1]))
  checkmate::assertList(newdata)
  if (!isTRUE(object$baseline)) {
    if (sum(apply(sapply(object$fdata, dim), 1, diff)) != 0) {
      stop("Error: objects in fdata have different dimensions")
    }
  } else {
    if (sum(diff(sapply(object$fdata, function(i) length(i)))) != 0) {
      stop("Error: objects in fdata have different dimensions")
    }
  }
  if (!isTRUE(object$baseline)) {
    if (sum(apply(sapply(newdata, dim), 1, diff)) != 0) {
      stop("Error: objects in fdata have different dimensions")
    }
  } else {
    if (sum(diff(sapply(newdata, function(i) length(i)))) != 0) {
      stop("Error: objects in fdata have different dimensions")
    }
  }
  checkmate::assertChoice(predict.type, choices = c("response", "prob"))
  if (object$nderiv != 0) {
    if (isTRUE(object$baseline)) {
      object$fdata <- plyr::alply(cbind(1:length(object$fdata)), 1, function(i) lapply(object$fdata[[i]], function(j) diff(j, lag = object$nderiv)))
      newdata <- plyr::alply(cbind(1:length(newdata)), 1, function(i) lapply(newdata[[i]], function(j) diff(j, lag = object$nderiv)))
    } else {
      object$fdata <- lapply(object$fdata, function(x) apply(x, 2, function(y) base::diff(y, lag = object$nderiv)))
      newdata <- lapply(newdata, function(x) apply(x, 2, function(y) base::diff(y, lag = object$nderiv)))
    }
  }

  if (is.null(object$mdist)) {
    mdist <- do.call(CMFold::compmDistMat, list(
      x = object$fdata,
      y = newdata,
      method = object$metric,
      measure = object$measure,
      baseline = object$baseline,
      weight = object$weight, # add condition for checking and so
      timestamps = object$timestamps,
      reset_timestamps = object$reset_timestamps,
      resample = object$resample,
      flip_threshold = object$flip_threshold,
      hover_threshold = object$hover_threshold,
      parallel = object$parallel,
      cl = object$cl,
      diag = object$diag,
      upper = object$upper
    ))
  } else {
    mdist <- msubset
  }
  # source(paste(paste("m","classiTies",sep=""),"R",sep="."))
  if (ncol(newdata[[1]]) == 1) { # this should be incorporated in classiKernel
    nn.list <- mclassiTies(x = mdist, knn = object$knn)
    if (predict.type == "response") {
      result <- (which.max(table(object$classes[nn.list$trajectories])) - 1)
    } else {
      result <- table(object$classes[nn.list$trajectories]) / length(object$classes[nn.list$trajectories])
    }
  } else {
    nn.list <- apply(mdist, 1, function(x) mclassiTies(x = x, knn = object$knn))
    if (predict.type == "response") {
      resultTable <- sapply(nn.list, function(x) table(factor(object$classes[x$trajectories], levels = levels(object$classes))))
      result <- apply(resultTable, 2, function(x) ifelse(length(unique(x)) == 1, sample(levels(object$classes), 1), which.max(x) - 1))
      result <- factor(result)
      # names(result)=colnames(newdata[[1]])
    } else {
      result <- t(sapply(nn.list, function(x) table(factor(object$classes[x$trajectories], levels = levels(object$classes)))))
      result <- t(apply(result, 1, function(x) x / sum(x)))
      # rownames(result)=colnames(newdata[[1]])}}
    }
  }
  return(result)
}
