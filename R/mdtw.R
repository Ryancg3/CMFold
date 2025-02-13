#' Dynamic Time Warping distance between multivariate functions
#'
#' Computes the Dynamic Time Warping distance for all pairs of \eqn{m}-dimensional functions.
#'
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @param ... additional arguments that can be passed to \code{\link[dtw:dtw]{dtw::dtw}}
#' @details Applies the \code{\link[dtw:dtw]{dtw::dtw}} function  to all pairs of \eqn{m}-dimensional functions.
#' By default, the Euclidean Distance between optimally aligned \eqn{m}-dimensional functions is computed.
#' Other options and parameters that can be passed to \code{\link[dtw:dtw]{dtw::dtw}} (such as the use of the Manhattan distance) can be defined in the additional parameters \code{...}
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional Dynamic Time Warp distances.
#' @seealso See \code{\link[parallel:makeCluster]} \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#' @inherit dtw::dtw return references
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mdtw(data, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mdtw(data, parallel = FALSE, cl = NULL)
#'
#' @export
mdtw = function(data, parallel=FALSE,cl=NULL, ...)
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
  data_array= array(unlist(data), dim=c(t,n,d))    # convert list into 3d array
  combinations = expandgrid2(matrix(rep(1:n,times=2),ncol=2)) #posible combinations between individuals : (1,1); (1,2) ....

  if(parallel==FALSE){
    return(array(apply(combinations,
                       1,
                       function(x) dtw::dtw(data_array[,x[1],], #function uses pracma v1.9.9 package
                                            data_array[,x[2],], ...)$distance),
                 dim=c(n,n)
    )
    )
  }else{
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("expandgrid2","data", "data_array"),
                            envir=environment())
    parallel::clusterEvalQ(cl, library(dtw))
    return(array(parallel::parApply(cl=cl,X=combinations,
                                    MARGIN=1,
                                    function(x) dtw::dtw(data_array[,x[1],], #function uses pracma v1.9.9 package
                                                        data_array[,x[2],], ...)$distance),
                 dim=c(n,n)
    )
    )

  }
}
