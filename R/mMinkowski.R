#' Minkowski distance between multivariate functions
#'
#' Computes the Minkowski distance for all pairs of \eqn{m}-dimensional functions.
#' For a single pair of functions, the present R function computes the sum of Minkowski distances
#' between the function values at equal time points.
#'
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param p integer to specify which p-norm to compute
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @details For each pair of functions f and g, the present R function computes: \eqn{sum t [Minkowski_distance(f(t), g(t))]}
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional Minkowski distances.
#' @seealso See \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, and \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mminkowski(data, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mminkowski(data, parallel = FALSE, cl = NULL)
#'
#' @export

mMinkowski=function(data,p=2,parallel=FALSE,cl=NULL)
{
  checkmate::assertList(data)
  checkmate::assertNumeric(p,lower=1,upper=Inf)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    checkmate::assertNumeric(cl,lower=1)}
  d=length(data)
  if(sum(apply(sapply(data,dim),1,diff))!=0){
    stop("Error: objects in fdata have different dimensions")}
  if(sum(sapply(data,function(x)sum(is.na(x))))!=0){
    warning("data have missing values; some distances cannot be computed.")}
  n=unique(sapply(data,ncol))
  if(parallel==FALSE)
  {
    return(apply(matrix(Reduce("+",lapply(lapply(data,function(x) proxy::dist(t(x),t(x),"Minkowski",p=p)),function(x) x^p)),ncol=n,nrow=n),1,function(x)x^{1/p}))
  } else {
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("data"))
    return(parallel::parApply(cl,matrix(Reduce("+",parallel::parLapply(cl,parallel::parLapply(cl,data,function(x) proxy::dist(t(x),t(x),"Minkowski",p=p)),function(x) x^p)),ncol=n,nrow=n),1,function(x)x^{1/p}))
    parallel::stopCluster(cl)}
}

