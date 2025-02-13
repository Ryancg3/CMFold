#' Global Minimum distance between multivariate functions
#'
#' Computes the Global Minimum distance for all pairs of \eqn{m}-dimensional functions. For a single pair of functions,
#' the present R function returns the minimum euclidean distance between the function values at equal time points.
#'
#' @param data a matrix that stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @return Returns a square and symmetric \eqn{n x n} matrix of \eqn{m}-dimensional global minimum distances.
#' @details For each pair of functions f and g, the present R function computes: \eqn{min t [Euclidean_Distance(f(t), g(t))]}
#' @seealso See \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}, and \code{\link[proxy:dist]{dist}} from \href{https://cran.r-project.org/web/packages/proxy/index.html}{proxy}
#' @examples
#' ## 2-dimensional functions
#'
#' x = replicate(4, rnorm(100, 0, 3))
#' y = replicate(4, rnorm(100, 3, 1))
#' data = list(x, y)
#' mglobmin(data, parallel = FALSE, cl = NULL)
#'
#' ## 3-dimensional functions
#'
#' z = replicate(4, rpois(100, 2))
#' data = list(x, y, z)
#' mglobmin(data, parallel = FALSE, cl = NULL)
#'
#' @export
mglobMin3=function(data,parallel=FALSE,cl=NULL)
{
  checkmate::assertMatrix(data)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    checkmate::assertNumeric(cl,lower=1)}
  if(sum(sapply(data,function(x)sum(is.na(x))))!=0){
    warning("data have missing values; some distances cannot be computed.")}
  n=ncol(data)
  if(parallel==FALSE){
    Mins= Rfast::colMins(data, value =TRUE) # returns minimum per column
    combs_lower_tri = utils::combn(1:n, 2, simplify=FALSE) #combinations for indexes of upper triangular matrix
    lower_tri_res = sapply(combs_lower_tri,function(x) abs(Mins[x[[1]]]-Mins[x[[2]]]) ) # compute results for upper triangular distance matrix
    
    } else {
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("Mins"), envir=environment())
    
    #compute results of upper/lower triangle of distance matrix
    lower_tri_res = parallel::parSapply(cl, 
                                        combs_lower_tri,
                                        function(x) abs(Mins[x[[1]]]-Mins[x[[2]]]) ) # compute results for upper triangular distance matrix
    
     parallel::stopCluster(cl)
     }
  
  #format results
  result_matrix = matrix(NA, nrow = n, ncol = n)
  result_matrix[lower.tri(result_matrix)] = lower_tri_res
  result_matrix = t(result_matrix)
  result_matrix[lower.tri(result_matrix)] = lower_tri_res
  diag(result_matrix)= rep(0, n)
  
  return(result_matrix)
}


#' @export

mglobMin2=function(data,parallel=FALSE,cl=NULL)
{
  checkmate::assertList(data)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    checkmate::assertNumeric(cl,lower=1)}
  d=length(data)
  if(d != 2){stop("Error: objects in data do not have 2 dimensions")}
  if(sum(apply(sapply(data,dim),1,diff))!=0){
    stop("Error: objects in data have different dimensions")}
  if(sum(sapply(data,function(x)sum(is.na(x))))!=0){
    warning("data have missing values; some distances cannot be computed.")}
  n=unique(sapply(data,ncol))
  if(parallel==FALSE){
    return(abs(mglobMin3(data[[1]])-mglobMin3(data[[2]])))
   
  } else {
    return(abs(mglobMin3(data[[1]], parallel = parallel, cl = cl)-mglobMin3(data[[2]], parallel = parallel, cl = cl )))
  }
}



