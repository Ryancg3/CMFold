#' Distance matrix among \eqn{m}-dimensional functions / trajectories
#'
#' The present function computes the distance matrix of one set of \eqn{n} different \eqn{m}-dimensional functions to another set of \eqn{k} different \eqn{m}-dimensional functions. The distances for all combinations of functions in the first set with functions of the second set are returned.
#' Different distance measures are available and can be set by \code{method}.
#'
#' @param x \code{list} of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param y \code{list} of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{k}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param method \code{string} to specify the distance measure to compute. Defaults to "Euclidean". Can be set to: ...
#' @param measure \code{string} to specify the measure to be computed with the  mousetrap package (can only be used instead of \code{"method"}).
#' The complete list of available measures can be retrieved with the function \code{\link[compmDistMat:mmeasuresChoices]{mmeasuresChoices}}.
#' For a detailed explanation of the different measures see \code{\link[mousetrap:mt_measures]{mousetrap:mt_measures}}. Can only be used for 2D data.
#' @param baseline is a binary variable indicating whether the distance matrix should be corrected by the baseline characteristics
#' @param weight OTHER PARAMETERS OF MOUSETRAP TO BE FINISHED
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @param diag is a logical value indicating whether the diagonal of the distance matrix should be printed. Only applicable if distances of one set of functions with itself is computed (\code{isnull(y)=TRUE}). If TRUE, diagonal values are NA.
#' @param upper is a logical value indicating whether the upper triangle of the matrix should be printed. Only applicable if distances of one set of functions with itself is computed (\code{isnull(y)=TRUE}). If TRUE, upper triangle values are NA.
#' @return The present function returns a rectangular matrix of size \eqn{k*n} containing the distances of
#' all \eqn{k} functions in \code{y} with all n functions/trajectories in \code{x}. If \code{y} is left unspecified, the function returns the distances of the functions in \code{x} with itself (a symmetric \eqn{n x n} matrix).
#'
#'
#' @seealso See \code{\link[parallel:makeCluster]} \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}
#' @examples
#' ## 2-dimensional functions
#'
#' xdim1 <- replicate(4, rnorm(100, 0, 3))
#' xdim2 <- replicate(4, rnorm(100, 3, 1))
#'
#' ydim1 <- replicate(2, rnorm(100, 0, 3))
#' ydim2 <- replicate(2, rnorm(100, 3, 1))
#'
#' x <- list(xdim1, xdim2)
#' y <- list(ydim1, ydim2)
#' compmDistMat(x = x, y = y, method = "Euclidean", parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#' xdim1 <- replicate(4, rnorm(100, 0, 3))
#' xdim2 <- replicate(4, rnorm(100, 3, 1))
#' xdim3 <- replicate(4, rnorm(100, 5, 1))
#'
#' ydim1 <- replicate(2, rnorm(100, 0, 3))
#' ydim2 <- replicate(2, rnorm(100, 3, 1))
#' ydim3 <- replicate(2, rnorm(100, 5, 1))
#'
#' x <- list(xdim1, xdim2, xdim3)
#' y <- list(ydim1, ydim2, ydim3)
#' compmDistMat(x = x, y = y, method = "Euclidean", parallel = FALSE, cl = NULL)
#'

#' compmDistMat: computes the distance (given "method") among n-dimensional vector(s)
#'
#' x is a list object with the components/coordinates (e.g., as matrices with columns as functions) of the n-dimensional functions (to fit the model)
#' y is a list object with the components/coordinates (e.g., as matrices with columns as functions) of the n-dimensional functions (to predict)
#' method is the distance measure to be used (similar to stat:dist)
#' measure is a logical value (defaul=FALSE). If TRUE, mtmeasures() function is called and measures are computed with mousetrap package
#' baseline is a binary variable indicating whether the distance matrix should be corrected by the baseline characteristics
#' diag is a logical value indicating whether the diagonal of the distance matrix should be printed (similar to stat:dist)
#' upper is a logical value indicating whether the upper triangle of the matrix should be printed (similar to stat:dist)
#' @export
compmDistMat <- function(x,
                         y = NULL,
                         method = "Euclidean",
                         measure = NULL,
                         timestamps = NULL,
                         reset_timestamps = FALSE,
                         resample = TRUE,
                         flip_threshold = 0,
                         hover_threshold = 2000,
                         baseline = FALSE,
                         weight = NULL, # add condition for checking and so
                         parallel = FALSE,
                         cl = NULL,
                         diag = TRUE,
                         upper = TRUE, ...) {
  # Block 1: overall checks
  checkmate::assertList(x)
  data <- x
  if (!is.null(timestamps)) {
    checkmate::assertList(timestamps)
  }
  checkmate::assertLogical(baseline)
  if (!isTRUE(baseline)) {
    d <- length(data)
  } else {
    d <- length(data[[1]])
  }
  checkmate::assertLogical(diag)
  checkmate::assertLogical(upper)
  checkmate::assertLogical(reset_timestamps)
  checkmate::assertLogical(resample)
  checkmate::assertNumeric(flip_threshold, lower = 0, len = 1L)
  checkmate::assertNumeric(hover_threshold, lower = 0, len = 1L)
  checkmate::assertLogical(parallel)
  if (!is.null(cl)) {
    if (!isTRUE(parallel)) {
      stop("Error: parallel should be TRUE")
    }
    checkmate::assertNumeric(cl, lower = 1)
  }

  # Block 2: list of objects (x: if there is not testing set, x and y if there is a testing set)
  if (!is.null(y)) {
    checkmate::assertList(y)
    if (!(length(x) == length(y))) { # check if the components have the same number of individuals and time-points
      stop("Error: x and y should have the same length")
    }
    allist <- list(x, y)
    if (isTRUE(baseline)) {
      data <- plyr::alply(cbind(1:d), 1, function(i) plyr::alply(cbind(1:length(allist[[1]])), 1, function(j) do.call(cbind, do.call(cbind, do.call(cbind, allist)[j, ])[i, ])))
    } else {
      data <- plyr::alply(cbind(1:d), 1, function(i) do.call(cbind, do.call(cbind, allist)[i, ]))
    }
  }

  # Block 3: compute the distance depending of method or measure
  if (!is.null(method)) {
    checkmate::assertNull(measure)
    # source(paste(paste(paste("m","metric",sep=""),"Choices",sep=""),"R",sep="."))
    checkmate::assertChoice(method, choices = mmetricChoices())
    if (method %in% unlist(sapply(proxy::pr_DB$get_entries()[c("Euclidean", "Manhattan", "Minkowski")], function(x) x$names))) {
      method <- names(sapply(sapply(proxy::pr_DB$get_entries(), function(x) x$names), function(x) sum(x == method))[sapply(sapply(proxy::pr_DB$get_entries(), function(x) x$names), function(x) sum(x == method)) == 1])
    }
    # source(paste(paste("m",method,sep=""),"R",sep="."))
    if (isTRUE(baseline)) {
      mdist <- plyr::alply(cbind(1:length(allist[[1]])), 1, function(i) do.call(paste("m", method, sep = ""), list(do.call(cbind, data)[i, ], parallel = parallel, cl = cl)))
      if (is.null(weight)) {
        weight <- rep(1 / length(allist), length(allist))
      }
      mdist <- plyr::alply(cbind(1:length(weight)), 1, function(j) mdist[[j]] * weight[j])
      mdist <- Reduce("+", mdist)
    } else {
      mdist <- do.call(paste("m", method, sep = ""), list(data, parallel = parallel, cl = cl))
    }
  } else {
    checkmate::assertNull(method)
    # source(paste(paste(paste("m","measures",sep=""),"Choices",sep=""),"R",sep="."))
    checkmate::assertChoice(measure, choices = mmeasuresChoices())
    # source(paste(paste("m","measures",sep=""),"R",sep="."))
    if (isTRUE(baseline)) {
      mtdata <- plyr::alply(cbind(1:length(allist[[1]])), 1, function(i) mmeasures(mtdata = data[[i]], timestamps = timestamps[[i]], reset_timestamps = reset_timestamps, resample = resample, flip_threshold = flip_threshold, hover_threshold = hover_threshold))
      if (measure == "flips") {
        mtdata <- lapply(mtdata, function(x) list(x = x[, "xpos_flips"], y = x[, "ypos_flips"]))
      } else {
        mtdata <- lapply(mtdata, function(x) list(x = x[, measure]))
      }
      # source(paste(paste(paste("m","dist",sep=""),"Measures",sep=""),"R",sep="."))
      mdist <- lapply(mtdata, function(i) mdistMeasures(i, measure))
      if (is.null(weight)) {
        weight <- rep(1 / length(allist), length(allist))
      }
      mdist <- plyr::alply(cbind(1:length(weight)), 1, function(j) mdist[[j]] * weight[j])
      mdist <- Reduce("+", mdist)
    } else {
      mtdata <- mmeasures(mtdata = data, timestamps = timestamps, reset_timestamps = reset_timestamps, resample = resample, flip_threshold = flip_threshold, hover_threshold = hover_threshold)
      # source(paste(paste(paste("m","dist",sep=""),"Measures",sep=""),"R",sep="."))
      if (measure == "flips") {
        mdist <- do.call(mdistMeasures, list(list(xlist = cbind(mtdata[, "xpos_flips"]), ylist = cbind(mtdata[, "ypos_flips"])), measure))
      } else {
        mdist <- do.call(mdistMeasures, list(list(mtdata[, measure]), measure))
      }
    }
  }

  # Block 4: prepare the output
  if (upper == TRUE) {
    if (diag == FALSE) {
      diag(mdist) <- NA
    }
  }
  if (upper == FALSE) {
    mdist[upper.tri(mdist, diag = !diag)] <- NA
  }
  rownames(mdist) <- paste("x", 1:nrow(mdist), sep = "-")
  colnames(mdist) <- paste("x", 1:nrow(mdist), sep = "-")
  if (!is.null(y)) {
    if (isTRUE(baseline)) {
      mdist <- mdist[(ncol(x[[1]][[1]]) + 1):(ncol(mdist)), 1:ncol(x[[1]][[1]])]
      if (ncol(y[[1]][[1]]) != 1) {
        rownames(mdist) <- paste("y", 1:nrow(mdist), sep = "-")
      }
    } else {
      mdist <- mdist[(ncol(x[[1]]) + 1):(ncol(mdist)), 1:ncol(x[[1]])]
      if (ncol(y[[1]]) != 1) {
        rownames(mdist) <- paste("y", 1:nrow(mdist), sep = "-")
      }
    }
  }

  return(mdist)
}

# TO DO(s):
# NAs -> distances give error if missing. Not measure (Amanda: check this)
