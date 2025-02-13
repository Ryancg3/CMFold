#' mclassiOnested
#'
#' task: define a task with makeClassifTask
#' data: data set with: id, dimension, trajectories as rows and condition
#' target: condition
#' knn: value for nearest neighbor
#' par.vals: hyperparameters
#' cv: nested loop (model performance) is done given k-fold cross-validation (cv=k)
#' M: number of iterations in the inner loop (subsampling)
#' weight: subsampling is used to tuned parameters. "weight" is a vector with weights for training and testing data sets
#' @export
mclassiOnested <- function(cl,
                           task,
                           knn,
                           id,
                           par.vals,
                           cv = 5,
                           M = 100,
                           weight) {
    osubset <- sample(cut(1:task$task.desc$size, breaks = cv, labels = 1:cv))
    list.osubset <- plyr::alply(cbind(1:cv), 1, function(i) (1:task$task.desc$size)[osubset == i])
    list.isubset <- plyr::alply(cbind(1:cv), 1, function(j) (1:task$task.desc$size)[!((1:task$task.desc$size) %in% list.osubset[[j]])])
    innerLoop <- replicate(M, apply(data.grid, 1, function(z) {
        sapply(list.isubset, function(k) {
            mclassiInested(
                cl = cl, task = task, id = id, knn = z[1:3], par.vals = par.vals, subset = k, weight = weight,
                super.learner = "boosting", par.super.learner = z[4:6]
            )
        })
    }))

    a <- apply(data.grid, 1, function(z) {
        sapply(list.isubset, function(k) {
            mclassiInested(
                cl = cl, task = task, id = id, knn = z[1:2], par.vals = par.vals, subset = k, weight = weight,
                super.learner = "boosting", par.super.learner = z[3:5]
            )
        })
    })

    M <- 1
    data.grid <- expand.grid(1:45, 1:20, seq(500, 4500, 1000), 1:length(id), 0.01)

    innerLoop <- apply(innerLoop, 1:2, mean)
    innerLoop <- cbind(innerLoop, 1:cv)
    innerLoopExtract <- plyr::alply(innerLoop, 1, function(x) innerResultsExtract(x))
    outerLoopExtract <- sapply(innerLoopExtract, function(x) mclassiPerf(cl = cl, task = task, knn = x$knn, par.vals, subset = list.isubset[[x$sample]]))
    outerLoopExtract <- cbind(set = 1:cv, accuracy = outerLoopExtract, knn = sapply(innerLoopExtract, function(x) x$knn))
    return(outerLoopExtract)
}


#' mclassiInested
#'
#' task: define a task with makeClassifTask
#' data: data set with: id, dimension, trajectories as rows and condition
#' target: condition
#' knn: value for nearest neighbor
#' par.vals: hyperparameters
#' subset: ith inner sample
#' weight: subsampling is used to tuned parameters. "weight" is a vector with weights for training and testing data sets
#' @export
#'

cl <- "classif.mclassiKnn"
fdata <- lapply(fdata, as.matrix)
Mdist_1 <- mManhattan(fdata, parallel = TRUE, cl = 8)
Mdist_2 <- mEuclidean(fdata, parallel = TRUE, cl = 8)
# Mdist_3=mglobMax(fdata,parallel=TRUE,cl=8)
task <- makeClassifTask(data = dataApp4, target = "condition")
id <- c("L1", "L2")
knn <- c(4, 4)
par.vals <- list(
    list(metric = "L1", mdist = Mdist_1, predict.type = "prob"),
    list(metric = "L2", mdist = Mdist_2, predict.type = "prob")
)
subset <- list.isubset[[1]]

test <- mclassiInested(
    cl = "classif.mclassiKnn",
    task = makeClassifTask(data = dataApp4, target = "condition"),
    id = c("L1", "L2", "globMax"),
    par.vals = list(
        list(metric = "L1", mdist = Mdist_1, predict.type = "prob"),
        list(metric = "L2", mdist = Mdist_2, predict.type = "prob"),
        list(metric = "globMax", mdist = Mdist_3, predict.type = "prob")
    ),
    knn = c(4, 4, 4),
    subset = list.isubset[[1]],
    weight = NULL,
    super.learner = "boosting",
    par.super.learner = c(300, 1, 0.01)
)

mclassiInested <- function(cl,
                           task,
                           id,
                           par.vals,
                           knn = 1L,
                           subset,
                           weight = NULL,
                           super.learner,
                           par.super.learner) {
    if (is.null(task)) {
        stop("Error: task is NULL")
    }
    if (length(id) == 1) {
        learner <- makeLearner(
            cl = cl,
            id = id,
            par.vals = list(
                metric = par.vals$metric,
                mdist = par.vals$mdist[((1:task$task.desc$size) %in% subset), ((1:task$task.desc$size) %in% subset)],
                knn = knn
            )
        )
    } else {
        learner <- mclassiStackLearner(task = task, id = id, knn = knn, kernel = NULL, par.vals = par.vals, subset = subset)
    }
    subtaskTrain <- subsetTask(task, subset = subset)
    isubset <- sample(c(TRUE, FALSE), size = length(subset), replace = TRUE, prob = weight)
    isubset <- c(1:length(isubset))[isubset]
    if (length(id) == 1) {
        model <- train(learner = learner, task = subtaskTrain, subset = isubset)
    } else {
        model <- mclassiStackTrain(learner, task = task, subset = isubset)
    }
    subtaskTest <- subsetTask(subtaskTrain, subset = c(1:length(subset))[!(c(1:length(subset)) %in% isubset)])
    if (length(id) == 1) {
        pred <- predict(model, task = subtaskTest)
        rreturn <- sum(diag(table(pred$data[, -1]))) / sum(table(pred$data[, -1]))
    } else {
        pred <- mclassiStackTest(learner, model, subtaskTest)
        rreturn <- mclassiStack(model, pred, par.super.learner = par.super.learner, super.learner = super.learner)
        rreturn <- rreturn$y
    }
    return(rreturn)
}

#' mclassiPerf
#'
#' cl: classifier
#' task: define a task with makeClassifTask
#' data: data set with: id, dimension, trajectories as rows and condition
#' target: condition
#' knn: value for nearest neighbor
#' par.vals: hyperparameters
#' subset: ith inner sample
#' #'@export
mclassiPerf <- function(cl,
                        task,
                        knn,
                        par.vals,
                        subset) {
    learnerPerm <- makeLearner(
        cl = cl,
        id = par.vals$metric,
        par.vals = list(
            metric = par.vals$metric,
            mdist = par.vals$mdist,
            knn = knn
        )
    )
    modelPerm <- train(learner = learnerPerm, task = task, subset = subset)
    pred <- predict(modelPerm, task = task, subset = (1:task$task.desc$size)[!((1:task$task.desc$size) %in% subset)])
    return(sum(diag(table(pred$data[, -1]))) / sum(table(pred$data[, -1])))
}

#' innerResultsExtract
#'
#' @export
innerResultsExtract <- function(x) {
    list(
        sample = x[length(x)],
        knn = which.max(x[-length(x)]),
        accuracy = x[which.max(x[-length(x)])]
    )
}
