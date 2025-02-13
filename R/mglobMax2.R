#' Global Maximum distance between multivariate functions
#'
#' Computes the Global Maximum distance for all pairs of \eqn{m}-dimensional functions. For a single pair of functions,
#' the present R function returns the maximum euclidean distance between the function values at equal time points.
#'
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional global maximum distances.
#' @details For each pair of functions f and g, the present R function computes: \eqn{max t [Euclidean_Distance(f(t), g(t))]}
#' @seealso See \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, and \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mglobmax(data, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mglobmax(data, parallel = FALSE, cl = NULL)
#'
#' @export
mglobMax2=function(data,parallel=FALSE,cl=NULL)
{
  checkmate::assertList(data)
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
  if(parallel==FALSE){
    return(matrix(apply(cbind(1:n),
                        1,
                        function(i) 
                          sqrt(colSums(
                            (t(sapply(data,
                                      function(x) Rfast::colMaxs(as.matrix(x),value=TRUE)))-t(sapply(data,function(x) Rfast::colMaxs(as.matrix(x),value=TRUE)))[,i])^2))),ncol=n,nrow=n,byrow=TRUE))
  
    
    
    } else {
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("data"))
    return(matrix(parallel::parRapply(cl=cl,cbind(1:n),function(i) sqrt(colSums((t(sapply(data,function(x) Rfast::colMaxs(as.matrix(x),value=TRUE)))-t(sapply(data,function(x) Rfast::colMaxs(as.matrix(x),value=TRUE)))[,i])^2))),ncol=n,nrow=n,byrow=TRUE))
    parallel::stopCluster(cl)}
}



