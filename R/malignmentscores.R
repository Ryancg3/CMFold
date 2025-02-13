#' Sequence alignment scores between multivariate functions / trajectories
#'
#' Computes Sequence Alignment scores for all pairs of \eqn{m}-dimensional functions.
#' Sequence alignment scores are usually computed for aligning sequences of nominal values. The present function extends 
#' the computation of such alignment scores to functions by splitting the function space into areas.
#' Thus, alignment scores of the sequence of locations can be calculated. 
#' 
#' @param data a list of \eqn{m} objects in matrix form. Each matrix stores a dimension of the set of functions, such that columns are individuals (\eqn{n}) and rows are discrete-time points (\eqn{t}). Functions' values should be of the same time points.
#' @param Keeprep logical value indicating whether adjacent duplicates should be kept or removed. 
#' For instance, if the function / trajectory has values within the same area for consecutive time stamps, should those consecutive duplicates be kept? Default is TRUE.
#' @param parallel logical value indicating whether computations should be parallelized. Default is FALSE. If TRUE, parallelization is conducted with \href{https://www.rdocumentation.org/packages/parallel}{parallel} package.
#' @param cl a cluster object created by \href{https://www.rdocumentation.org/packages/parallel}{parallel}. Default is NULL.
#' @param ... additional arguments that can be passed to \code{\link[Biostrings:pairwiseAlignment]{Biostrings::pairwiseAlignment}}
#' @details The present function translates each coordinate at each time point into an area.
#' The sequence of areas (encoded as characters) is then used in the \code{\link[Biostrings:pairwiseAlignment]{Biostrings::pairwiseAlignment}}
#' function. 
#' @return Returns a square and symmetric \eqn{n x n} matrix of alignment scores.
#' @seealso See \code{\link[parallel:makeCluster]} \code{\link[parallel:makeCluster]{makeCluster}}, \code{\link[parallel:clusterExport]{clusterExport}}, \code{\link[parallel:stopCluster]{stopCluster}}, \code{\link[parallel:parApply]{parApply}} and \code{\link[parallel:parLapply]{parLapply}} from  \href{https://www.rdocumentation.org/packages/parallel}{parallel}
#' @export

malignmentscore = function(data,#This function takes as input a list of data.frames (one data frame for each dimension)
                                                  #Columns contain data for different Individual 
                                                   #Rows contain data for different time stamps
                           Keeprep= TRUE ,# should consecutive locations in the same area be kept? or dropped?
                           parallel=FALSE,
                           cl=NULL,
                           ...){ #options for the Biostrings::pairwise alignment function
                                       
  ###################################################################################
  ##1. TESTS FOR CORRECT INPUT
  ###################################################################################
 
  checkmate::assertLogical(Keeprep)
  data = as.matrix(data)
  class(data) = "character"  

  ##################################################################################
  ## 2. Preprocess data: transform "areas" to character strings
  ##################################################################################
  
  #transform areas to characters (e.g "area1" --> "e")
  Area_encoded_as_String = unique(as.vector(data))
  Allowed_characters = c(intToUtf8(65:90, multiple=TRUE), intToUtf8(45:65, multiple=TRUE))
  if (length(Area_encoded_as_String) >47){print("The entered sequence has too many possible values. Maximum number of unique values is 47 ")}
  
  Area_Encodedascharacter = Allowed_characters[rank(Area_encoded_as_String)] #need be string and matrix
  
  

  
  
  #map area codes to characters in the data
  char_seq = as.list(data.frame(plyr::mapvalues(data,
                                   from = Area_encoded_as_String ,
                                   to= Area_Encodedascharacter,
                                   warn_missing = FALSE)
                         ,stringsAsFactors = FALSE))
  
  #delete adjacent duplicates aaabbbaaaccc --> abac (optional)
  if (Keeprep == FALSE){char_seq = lapply(char_seq, function(y) rle(y)[[2]])}
  
  # form string of characters for each individual (position "DNA")
  char_seq= sapply(char_seq, function(x) paste(x, collapse = ""), simplify= TRUE )

  #################################################################################
  #3 Implement Pairwise alignment criterias
  #################################################################################
  
  #compute list of indexes of distance matrix
  combs_lower_tri = combn(1:length(char_seq), 2, simplify=FALSE) #list of indices for uppe rtriangular distances matrix
  combs_diag = lapply(1:length(char_seq), function(y) c(y,y)) #list of indidices diagonal terms
  
  #specify substitution matrix for alignement score 
  subMatrix = matrix(rep(0, length(Area_Encodedascharacter)), nrow=length(Area_Encodedascharacter), ncol=length(Area_Encodedascharacter))
  diag(subMatrix) = rep(1, length(Area_Encodedascharacter))
  rownames(subMatrix)= Area_Encodedascharacter
  colnames(subMatrix)= Area_Encodedascharacter
  
  
  if(parallel==FALSE){

    #distances upper triangle part of distance matrix
    lower_tri_res = sapply(combs_lower_tri, 
                           function(y) Biostrings::pairwiseAlignment(char_seq[y[1]],
                                                                     char_seq[y[2]],
                                                                     substitutionMatrix = subMatrix,
                                                                     scoreOnly=TRUE,
                                                                     ...) 
                           ) 

    #distances upper triangle part of distance matrix
    diag_res = sapply(combs_diag, 
                      function(y) Biostrings::pairwiseAlignment(char_seq[y[1]],
                                                                char_seq[y[2]], 
                                                                substitutionMatrix = subMatrix ,
                                                                scoreOnly=TRUE,
                                                                ...) 
                      )

    } else { # TO FINISH
      
    cl=parallel::makeCluster(cl)
    parallel::clusterExport(cl,list("char_seq", "subMatrix"), envir=environment())
    
    lower_tri_res = parallel::parSapply(cl, 
                                       combs_lower_tri,
                                       function(y) Biostrings::pairwiseAlignment(char_seq[y[1]],
                                                                                 char_seq[y[2]],
                                                                                 substitutionMatrix = subMatrix ,
                                                                                 scoreOnly=TRUE,
                                                                                 ...) ) 
    
    diag_res = parallel::parSapply(cl, 
                                  combs_diag,
                                  function(y) Biostrings::pairwiseAlignment(char_seq[y[1]],
                                                                            char_seq[y[2]],
                                                                            substitutionMatrix = subMatrix ,
                                                                            scoreOnly=TRUE,
                                                                            ...) ) #...
    
    parallel::stopCluster(cl)
    }

  #arange distances into matrix 
  result_matrix = matrix(NA, nrow = length(char_seq), ncol = length(char_seq))
  result_matrix[lower.tri(result_matrix)] = lower_tri_res
  result_matrix = t(result_matrix)
  result_matrix[lower.tri(result_matrix)] = lower_tri_res
  diag(result_matrix)= diag_res
  
  result_matrix = -result_matrix

  return(result_matrix)
}
