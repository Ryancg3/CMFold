#' mclassiEnsemble
#' @importFrom mlr makeLearner
#' @importFrom mlr predictLearner
#' @importFrom mlr trainLearner
#' @importFrom mlr makeClassifTask
#' @importFrom mlr tuneParams


mclassiStackLearner <- function(task,
                                id,
                                knn,
                                kernel = NULL,
                                par.vals,
                                subset = NULL) {
  if (is.null(task)) {
    stop("Error: task is NULL")
  }
  if (length(id) == 1) {
    stop("Error: id should be greater than or equal to 2")
  }
  learner <- list()
  for (j in 1:length(id)) {
    if (is.null(subset)) {
      learner[[j]] <- makeLearner(cl = cl, id = id[j], par.vals = list(metric = par.vals[[j]]$metric, mdist = par.vals[[j]]$mdist , knn = knn[j], predict.type = par.vals[[j]]$predict.type))
    }
    else {
      learner[[j]] <- makeLearner(cl = cl, id = id[j], par.vals = list(metric = par.vals[[j]]$metric, mdist = par.vals[[j]]$mdist[((1:task$task.desc$size) %in% subset), ((1:task$task.desc$size) %in% subset)], knn = knn[j], predict.type = par.vals[[j]]$predict.type))
    }
  }
  class(learner) <- c("StackLearner")
  learner
  }

mclassiStackTrain <- function(learner,
                              task,
                              subset,
                              outer = FALSE) {
  model <- list()
  if (outer == TRUE) { #is this needed, i think both variants are the same. 
    for (j in 1:length(id)) {
      model[[j]] <- train(learner = learner[[j]], task = task, subset=subset)
  }
  }
  else {
    subtaskTrain <- subsetTask(task, subset = c(1:task$task.desc$size)[(c(1:task$task.desc$size) %in% subset)])
    for (j in 1:length(id)) {
      model[[j]] <- train(learner = learner[[j]], task = subtaskTrain)
  }
  }

  class(model) <- c("StackTrain")
  return(model)
}

mclassiStackTest <- function(learner, model, task) {
  if (class(model) != "StackTrain") {
    stop("Error: model object should be of class StackTrain")
  }
  pred <- list()
  for (j in 1:length(model)) {
    pred[[j]] <- predictLearner(.learner = learner[[j]], .model = model[[j]], .newdata = task$env$data[-ncol(task$env$data)])
  }
  class(pred) <- c("StackTest")
  predRes <- list()
  predRes[[1]] <- pred
  predRes[[2]] <- task$env$data[ncol(task$env$data)]
  names(predRes) <- c("probabilities", "condition")
  class(predRes) <- "StackTest"
  return(predRes)
}

mclassiStack <- function(model, pred, super.learner = "randomForest", par.super.learner) {
  if (class(model) != "StackTrain") {
    stop("Error: model object should be of class StackTrain")
  }
  if (class(pred) != "StackTest") {
    stop("Error: model object should be of class StackTest")
  }
  checkmate::assertChoice(super.learner, choices = c("randomForest", "boosting", "nnet"))
  leveloneData <- cbind(pred[[2]], sapply(pred[[1]], function(x) as.numeric(as.character(x))))
  colnames(leveloneData)[1] <- "response"
  colnames(leveloneData)[2:ncol(leveloneData)] <- paste("exp", 2:ncol(leveloneData) - 1, sep = "")
  if (super.learner == "randomForest") {
    learner.aux <- makeLearner(
      cl = "classif.randomForest",
      predict.type = "prob"
    )
    task.aux <- makeClassifTask(data = leveloneData, target = "response")
    ps.aux <- makeParamSet(
      makeDiscreteParam("ntree", values = par.super.learner[1]),
      makeDiscreteParam("mtry", values = par.super.learner[2])
    )
    cv.aux <- makeResampleDesc("CV", iters = 10)
    par.aux <- tuneParams(learner.aux,
      task = task.aux, resampling = cv.aux, par.set = ps.aux,
      control = makeTuneControlGrid(), measure = acc
    )
  } else if (super.learner == "boosting") {
    learner.aux <- makeLearner(
      cl = "classif.gbm",
      distribution = "bernoulli",
      predict.type = "prob"
    )
    task.aux <- makeClassifTask(data = leveloneData, target = "response")
    ps.aux <- makeParamSet(
      makeDiscreteParam("n.trees", values = par.super.learner[1]),
      makeDiscreteParam("interaction.depth", values = par.super.learner[2]),
      makeDiscreteParam("shrinkage", values = par.super.learner[3])
    )
    cv.aux <- makeResampleDesc("CV", iters = 10)
    par.aux <- tuneParams(learner.aux,
      task = task.aux, resampling = cv.aux, par.set = ps.aux,
      control = makeTuneControlRandom(maxit = 1), measure = acc
    )
  } else {
    learner.aux <- makeLearner(
      cl = "classif.nnet",
      predict.type = "prob",
      skip = TRUE
    )
    task.aux <- makeClassifTask(data = leveloneData, target = "response")
    ps.aux <- makeParamSet(
      makeDiscreteParam("size", values = par.super.learner[1]),
      makeDiscreteParam("maxit", values = par.super.learner[2]),
      makeDiscreteParam("decay", values = par.super.learner[3])
    )
    cv.aux <- makeResampleDesc("CV", iters = 10)
    par.aux <- tuneParams(learner.aux,
      task = task.aux, resampling = cv.aux, par.set = ps.aux,
      control = makeTuneControlRandom(maxit = 1), measure = acc
    )
  }
  return(par.aux)
}
