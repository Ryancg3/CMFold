#' Distance between measures of 2-Dimensional trajectories.
#' 
#' The mousetrap package enables the calculation of certain measures of mouse trajectories.
#' The present function enables the computation of "distances" between such measures.
#'
#' @param data \code{list} containing one vector (or two if \code{measure="flips"})
#' of measures computed by the function \code{\link[classiMultiFunc:mmeasures]{mmeasures}}
#' @param measure \code{string} to specify the measure contained in \code{data}. 
#' The complete list of available measures can be retrieved with the function \code{\link[compmDistMat:mmeasuresChoices]{mmeasuresChoices}}. 
#' For a detailed explanation of the different measures see \code{\link[mousetrap:mt_measures]{mousetrap:mt_measures}}.
#' @examples 
#' #Example 1: measure != "flips"
#' data = list(c(1,2,3))
#' mdistMeasures(data=data, measure ="ypos_max")
#' 
#' #Example 2: measure = "flips"
#' data = list(c(1,2,3), c(4,5,6))
#' mdistMeasures(data=data, measure ="flips")
#'
#' return
#' @export
mdistMeasures=function(data,measure)
{
  checkmate::assertList(data)
  d=length(data)
  n=unique(sapply(data,length))
  checkmate::assertCharacter(measure)
  checkmate::assertChoice(measure,choice=mmeasuresChoices())
  if(measure=="flips"){
    if(sum(apply(sapply(data,function(x) dim(cbind(x))),1,diff))!=0){
      stop("Error: objects in data have different dimensions")}
    return(matrix(sqrt(Reduce("+",lapply(data,function(x)(proxy::dist(x,x,method="L2"))^2))),ncol=n,nrow=n))}
  if(measure %in% mmeasuresChoices()){
    if(d!=1){
      stop("Error: data should be a list of length 1")}
    return(lapply(data,function(x) apply(cbind(1:n),1,function(i) sqrt((x-x[i])^2)))[[1]])}
}

