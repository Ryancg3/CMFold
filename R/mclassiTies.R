#' mclassiTies
#'
#' Provides the positions of the indices of the k nearest trajectories. 
#' Selects randomly among ties for the \eqn{k}th nearest trajectory. 
#' 
#' @param x vector of distances of a new trajectory to all training trajectories. 
#' @param knn  number of nearest neighbours
#' 
#' @examples
#' ## distance of new data to the 10 trajectories in the training data:
#' dist_to_train_fucntions = c(1,2,2,2,2,2,5,5,5,6)
#' mclassiTies(x = dist_to_train_fucntions, knn=3)
#' @export


mclassiTies=function(x,knn,...)
{
  checkmate::assertVector(x)
  checkmate::assertNumeric(knn,min.len=1,lower=1)
  xrange=data.table::frank(x,ties.method="dense")
  tableTies=table(xrange[xrange<=knn])
  if(sum(cumsum(tableTies)==knn)!=0){
    if(which((cumsum(tableTies)==knn)==TRUE)==knn){
      trajTies=which(xrange<=knn) # no ties
      i=0
      } else {
        trajTies=which(xrange %in% as.numeric(names(tableTies[cumsum(tableTies)<=knn])))
        i=1} # ties example 1
    } else {
      if(length(tableTies)==1){
        trajTies=sample(which(xrange==names(tableTies[1])),knn,replace=FALSE)  # ties example 2
        i=1
        } else {
          trajNoTies=which(xrange %in% names(tableTies[cumsum(tableTies)<=knn])) # ties example 3
          trajTies=sample(which(xrange==names(tableTies[length(tableTies[cumsum(tableTies)<=knn])+1])),(knn-sum(tableTies[cumsum(tableTies)<=knn])),replace=FALSE)
          trajTies=c(trajNoTies,trajTies)
          i=1}
      }

  listTrajTies=list(x=x[trajTies],
                    trajectories=trajTies,
                    knn=knn,
                    ties=ifelse(i==1,"yes","no"))
  class(listTrajTies)="mclassiTies"
  return(listTrajTies)
}

#' @export
# print.mclassiTies
print.mclassiTies=function(x,...) {
  cat("\n")
  cat("\t mclassiTies object \n")
  cat("\n")
  if(length(x$x)>10){
    cat(" distance:",paste(round(sort(x$x)[1:10],3)),"[...]","\n")
    cat(" trajectories:",paste(x$trajectories[order(x$x)][1:10]),"[...]","\n")
  } else {
    cat(" distance:",paste(round(sort(x$x),3)),"\n")
    cat(" trajectories:",paste(x$trajectories[order(x$x)]),"\n")}
  cat(" knn:", x$knn, "\n")
  cat(" ties?:", x$ties, "\n")
  cat("\n")
}




