#' predict_mclassiKernel
#' @export

predict.mclassiKernel <- function(object, newdata = NULL, predict.type = "response") {
  # added to fit mlr format to the mclassifunc package
  d <- unique(newdata[, "dim"])
  newdata <- t(newdata[, -which(names(newdata) %in% c("id", "dim"))])
  newdata <- plyr::alply(
    cbind(1:d),
    1,
    function(i) data.frame(
      dim = rep(1:d, each = nrow(newdata) / d),
      newdata)
      [data.frame(dim = rep(1:d, each = nrow(newdata) / d), newdata)[, "dim"] == i, -1]
  )
  checkmate::assertList(newdata)
  # added to fit mlr format to the mclassifunc package

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
      object$fdata <- plyr::alply(
        cbind(1:length(object$fdata)), 
        1, 
        function(i) lapply(
          object$fdata[[i]], 
          function(j) diff(j, lag = object$nderiv)
          )
          )
      newdata <- plyr::alply(cbind(1:length(newdata)), 1, function(i) lapply(newdata[[i]], function(j) diff(j, lag = object$nderiv)))
    } else {
      object$fdata <- lapply(object$fdata, function(x) apply(x, 2, function(y) base::diff(y, lag = object$nderiv)))
      newdata <- lapply(newdata, function(x) apply(x, 2, function(y) base::diff(y, lag = object$nderiv)))
    }
  }

  mdist <- do.call(compmDistMat, list(
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
  print("HIIIIIII")
  #print(object)
  print(object$kernel)
  print(object$classes)
  print(mdist)

  mwn <- sapply(
    plyr::alply(
      mdist, 
      1, 
      function(x) do.call(object$kernel, list(x / object$h))
      ), 
      function(x) tapply(x, object$classes, sum)
      )
  mwd <- sapply(plyr::alply(mdist, 1, function(x) do.call(object$kernel, list(x / object$h))), sum)
  print(predict.type)
  if (predict.type == "response") {
    result <- factor(apply(apply(cbind(1:length(mwd)), 1, function(i) mwn[, i] / mwd[i]), 2, function(x) which.max(x) - 1), levels = levels(object$classes))
    names(result) <- rownames(mdist)
  } else {
    result <- t(apply(cbind(1:length(mwd)), 1, function(i) mwn[, i] / mwd[i]))
    rownames(result) <- rownames(mdist)
  }
  print(result)
  return(result)
}
