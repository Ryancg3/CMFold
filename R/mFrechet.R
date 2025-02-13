#'
#'@export
#'
#############################################################
# mFrechet: Frechet distance of n-dimensional trajectories #
#############################################################
# data is a list with  the components/coordinates/locations of the n-dimensional trajectories (columns: trajectories - rows: timestamps)

mFrechet=function(data,parallel=FALSE,cl=NULL,testLeash=-1)
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
  t=unique(sapply(data,nrow))
  d= length(data)
  
  # 3d matrix of data
  data_array= array(unlist(data), dim=c(t,n,d)) 
  
  #list of indices for upper rtriangular distances matrix
  combs_lower_tri = combn(1:n, 2, simplify=FALSE) 

  
  if(parallel==FALSE){
    #compute results of upper/lower triangle of distance matrix
    lower_tri_res = sapply(combs_lower_tri,
                           function(x) SimilarityMeasures::Frechet(data_array[,x[1],],
                                                                   data_array[,x[2],] ,
                                                                     testLeash=testLeash))
  } else {
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("data", "testLeash"), envir=environment())
    
    #compute results of upper/lower triangle of distance matrix
    lower_tri_res = parallel::parSapply(cl, 
                                        combs_lower_tri,
                                        function(x) SimilarityMeasures::Frechet(data_array[,x[1],],
                                                                                data_array[,x[2],] ,
                                                                                testLeash=testLeash))

    parallel::stopCluster(cl)
  }
  
  #arrange results into distance matrix
  result_matrix = matrix(NA, nrow = n, ncol = n)
  result_matrix[lower.tri(result_matrix)] = lower_tri_res
  result_matrix = t(result_matrix)
  result_matrix[lower.tri(result_matrix)] = lower_tri_res
  diag(result_matrix)= rep(0, n)
  return(result_matrix)
}


