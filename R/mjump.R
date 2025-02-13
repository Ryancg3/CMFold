#' Difference of the distance travelled between two time points of different multivariate functions
#'
#' For each \eqn{m}-dimensional function/trajectory the present R function computes the euclidean distance between the positions at two different time points (the "Jump"). The output matrix returns
#' the absolute value of the Jump difference for each pair of functions/trajectories.
#'
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param t1_quantile quantile of the time point at which the Jump starts (values in [0,1])
#' @param t2_quantile quantile of the time point at which the Jump end  (values in [0,1])
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @details For each pair of functions f and g, the present R function computes: \eqn{Abs(Euclidean_distance(f(t2)-f(t1)) - Euclidean_distance(g(t2)-g(t1))]}
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional "Jump" differences.
#' @seealso See \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, and \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mjump(data,t1_quantile=0.1, t2_quantile=0.8, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mjump(data,t1_quantile=0.1, t2_quantile=0.8, parallel = FALSE, cl = NULL)
#'
#' @export

mjump = function(data,t1_quantile,t2_quantile, parallel=FALSE,cl=NULL){
  checkmate::assertList(data)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    #checkmate::assertNumeric(cl,lower=1)
  }
  if(sum(apply(sapply(data,dim),1,diff))!=0){
    stop("Error: objects in data have different dimensions")}

  n=unique(sapply(data,ncol))
  t= nrow(data[[1]])
  n= ncol(data[[1]])
  d= length(data)

  t1_quantile = round(quantile(1:t, probs = t1_quantile))
  t2_quantile = round(quantile(1:t, probs = t2_quantile))

  data <- sqrt(Reduce("+",
                      lapply(
                        lapply(data, function(x) x[t2_quantile,]- x[t1_quantile,]),
                        function(x) x^2)
                      )
               )

  return(abs(outer(data, data, "-")))
}
