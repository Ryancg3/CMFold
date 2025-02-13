#' Hausdorff distance between multivariate functions
#'
#' Computes the Hausdorff distance for all pairs of \eqn{m}-dimensional functions.
#'
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @details Applies \code{\link[pracma:hausdorff_dist]{pracma::hausdorff_dist}} to all pairs of \eqn{m}-dimensional functions
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional Hausdorff distances.
#' @seealso See  \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#'
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mhausdorff(data, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mhausdorff(data, parallel = FALSE, cl = NULL)
#'
#' @export
mHausdorff = function(data,parallel=FALSE,cl=NULL)
{
  checkmate::assertList(data)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    #checkmate::assertNumeric(cl,lower=1)
    }
  if(sum(apply(sapply(data,dim),1,diff))!=0){
    stop("Error: objects in data have different dimensions")}

  expandgrid2=function(x){
    expand.grid(data.frame(x))}

n=unique(sapply(data,ncol))
t= nrow(data[[1]])
n= ncol(data[[1]])
d= length(data)
data_array= array(unlist(data), dim=c(t,n,d))
combinations = expandgrid2(matrix(rep(1:n,times=2),ncol=2))

if(parallel==FALSE){
  return(array(apply(combinations,1,function(x) pracma::hausdorff_dist(data_array[,x[1],],data_array[,x[2],])),dim=c(n,n)))
  } else {
  cl=parallel::makeCluster(cl)
  parallel::clusterExport(cl,list("expandgrid2","data", "data_array"),envir=environment())
  parallel::clusterEvalQ(cl,library(pracma))
  return(array(parallel::parApply(cl=cl,X=combinations,MARGIN=1,function(x) pracma::hausdorff_dist(data_array[,x[1],],data_array[,x[2],])),dim=c(n,n)))
  parallel::parallelStop()
  }
}


