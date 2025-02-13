#' Create mKNN Object
#'
#' The present function creates an object to specify the k-nearest-neighbor classification
#' for \eqn{n}-different observations of \eqn{m}-dimensional functionnal data.
#'
#'
#' @param classes  \code{factor} or \code{numeric}. A vector containing the true classes of the training data.
#' @param fdata the training covariates as a \code{list} of \eqn{m} objects in \code{matrix} form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param knn \code{integer} specifying the number of nearest neighbours considered in the knn algorithm.
#' @param metric \code{string} specifying the metric for knn. The complete list of available metrics is returned by
#' \code{mdistMeasures()}
#' @param nderiv \code{integer} number of derivatives of the trajectory.
#' @param measure is a logical value (defaul=FALSE). If TRUE, mtmeasures() function is called and measures are computed with mousetrap package
#' @param baseline is a binary variable indicating whether the distance matrix should be corrected by the baseline characteristics
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @param diag is a logical value indicating whether the diagonal of the distance matrix should be printed. Only applicable if distances of one set of functions with itself is computed (\code{isnull(y)=TRUE}). If TRUE, diagonal values are NA.
#' @param upper is a logical value indicating whether the upper triangle of the matrix should be printed. Only applicable if distances of one set of functions with itself is computed (\code{isnull(y)=TRUE}). If TRUE, upper triangle values are NA.
#' @return Returns a list containing the parameters and the training data for a mknn prediction. This list is used for prediction with the function \code{\link[classiMultiFuncTest:predict.mclassiKnn]{predict.mclassiKnn}}
#'
#' @seealso See \code{\link[parallel:makeCluster]} \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}
#' @examples
#'
#' ## 3-dimensional functions
#' #classes
#' classes = as.factor(c("Cat","Dog","Dog","Cat"))
#'
#' #fdata
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' z = replicate(4, rpois(100, 2))
#' fdata = list(x, y, z)
#'
#'
#' mclassiKnn(classes=classes, fdata = fdata, knn=3, nderiv = 0, cl = NULL)
#' @export

mclassiKnn=function(classes,
                    fdata,
                    mdist=NULL,
                    metric="Euclidean",
                    knn=1L,
                    nderiv=0L,
                    baseline=FALSE,
                    weight=NULL,
                    measure=NULL,
                    timestamps=NULL,
                    reset_timestamps=FALSE,
                    resample=TRUE,
                    flip_threshold=0,
                    hover_threshold=2000,
                    parallel=FALSE,
                    cl=NULL,
                    diag=FALSE,
                    upper=FALSE,...)
{
  if(is.numeric(classes)){
    classes=as.factor(classes)}
  checkmate::assertFactor(classes)
  checkmate::assertDataFrame(fdata)
  d=unique(fdata[,"dim"])
  fdata=t(fdata[,-which(names(fdata) %in% c("id","dim"))])
  fdata=plyr::alply(cbind(1:d),1,function(i) cbind(data.frame(dim=rep(1:d,each=nrow(fdata)/d),fdata)[data.frame(dim=rep(1:d,each=nrow(fdata)/d),fdata)[,"dim"]==i,-1]))
  checkmate::assertList(fdata)
  if(!isTRUE(baseline)){
    if(sum(apply(sapply(fdata,dim),1,diff))!=0){
      stop("Error: objects in fdata have different dimensions")}
    } else {
      if(sum(diff(sapply(fdata, function(i) length(i))))!=0){
        stop("Error: objects in fdata have different dimensions")}}
  checkmate::assertIntegerish(knn,lower=1L,len=1)
  checkmate::assertIntegerish(nderiv,lower=0L)
  if(!is.null(measure)){
    checkmate::assertNull(metric)
    checkmate::assertChoice(measure,choices=mmeasuresChoices())
  } else {
    checkmate::assertNull(measure)
    checkmate::assertCharacter(metric)
    checkmate::assertChoice(metric,choices=mmetricChoices())}
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    checkmate::assertNumeric(cl,lower=1)}
  nret=list(classes=classes,
            fdata=fdata,
            mdist=mdist,
            knn=knn,
            metric=metric,
            nderiv=nderiv,
            measure=measure,
            timestamps=timestamps,
            reset_timestamps=reset_timestamps,
            resample=resample,
            flip_threshold=flip_threshold,
            hover_threshold=hover_threshold,
            baseline=baseline,
            weight=weight,
            parallel=parallel,
            cl=cl,
            diag=diag,
            upper=upper)
  # class(nret)="mclassiKnn"
  return(nret)
}

# TO DO(s):
  # data transformation into funcional data
  # parallelization

