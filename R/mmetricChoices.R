#' mmetricChoices
#'@export
mmetricChoices=function(proxy.only=FALSE)
{
  proxy.list=proxy::pr_DB$get_entries()
  proxy_metric_names=c(unlist(sapply(proxy.list[c("Euclidean",
                                                  "Manhattan",
                                                  "Minkowski")],function(x)x$names)),
                        paste("supremum",1:2,sep=""),
                        paste("infimum" ,1:2,sep=""),
                       "mean","minimum","globMax","globMin","shortEuclidean","Frechet","Hausdorff","dtw")
  return(proxy_metric_names)
}


