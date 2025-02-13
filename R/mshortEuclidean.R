#' Short Euclidean distance between multivariate functions
#'
#' Computes the Short Euclidean distance for all pairs of \eqn{m}-dimensional functions.
#' For a single pair of functions, the Short Euclidean distance computes the sum of Euclidean distances
#' between the function values at equal time points within a predefined range.
#'
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @param t_inf_quantile lower quantile for the range of time points taken into consideration (values in [0,1])
#' @param t_sup_quantile upper quantile for the range of time points taken into consideration (values in [0,1])
#' @details For each pair of functions f and g, the present R function computes: \eqn{sum t [Euclidean_distance(f(t), g(t))]} for t in \eqn{[Total_time*t_inf_quantile, Total_time*t_sup_quantile]}
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional Supremum1 distances.
#' @seealso See \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, and \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mshortEuclidean(data, t_inf_quantile=0.2, t_sup_quantile= 0.8, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mshortEuclidean(data, t_inf_quantile=0.2, t_sup_quantile= 0.8, parallel = FALSE, cl = NULL)
#'
#' @export

mshortEuclidean=function(data,t_inf_quantile,t_sup_quantile,parallel=FALSE,cl=NULL)
{
  checkmate::assertList(data)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    checkmate::assertNumeric(cl,lower=1)}
  d=length(data)
  if(sum(apply(sapply(data,dim),1,diff))!=0){
    stop("Error: objects in data have different dimensions")}
  if(sum(sapply(data,function(x)sum(is.na(x))))!=0){
    warning("data have missing values; some distances cannot be computed.")}
  n=unique(sapply(data,ncol))
  checkmate::assertNumeric(t_inf_quantile,lower=0,upper=1,len=1L)
  checkmate::assertNumeric(t_sup_quantile,lower=0,upper=1,len=1L)
  stopifnot(t_inf_quantile<=t_sup_quantile)
  t_inf_quantile=round(quantile(1:nrow(data[[1]]),probs=t_inf_quantile))
  t_sup_quantile=round(quantile(1:nrow(data[[1]]),probs=t_sup_quantile))
  if(parallel==FALSE){
    return(ntEuclidean(lapply(data,function(i) i[t_inf_quantile:t_sup_quantile,])))
  } else {
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("data"))
    return(ntEuclidean(parallel::parLapply(cl,data,function(i) i[t_inf_quantile:t_sup_quantile,])))
    parallel::stopCluster(cl)}
}

