#'mclassiKernel:
#'
#'mclassiKernel: computes the distance (given "method") among n-dimensional vector(s) #
#'
#' @export
mclassiKernel=function(classes,
                       fdata,
                       metric="Euclidean",
                       nderiv= 0, #default added because of strange behavior of mlr defaults
                       kernel= "Ker.norm", #default added because of strange behavior of mlr defaults
                       h = 1,#default added because of strange behavior of mlr defaults
                       baseline=FALSE,
                       weight=NULL, 
                       measure=NULL,
                       timestamps=NULL,
                       reset_timestamps=FALSE,
                       resample=TRUE,
                       flip_threshold=0,
                       hover_threshold=2000,
                       parallel=FALSE,
                       cl=NULL,
                       diag=FALSE,
                       upper=FALSE,...) 
{
  if(is.numeric(classes)){
    classes=as.factor(classes)}
  #added for re-transforming data to a list
  d=unique(fdata[,"dim"])
  fdata=t(fdata[,-which(names(fdata) %in% c("id","dim"))])
  fdata=plyr::alply(cbind(1:d),
                    1,
                    function(i) data.frame(dim=rep(1:d,
                                                   each=nrow(fdata)/d)
                                           ,fdata)[data.frame(dim=rep(1:d,
                                                                      each=nrow(fdata)/d)
                                                              ,fdata)[,"dim"]==i,-1])
  #added for re-transforming data to a list
  
  checkmate::assertFactor(classes,len=ncol(fdata[[1]]))
  checkmate::assertList(fdata)
  if(!isTRUE(baseline)){
    if(sum(apply(sapply(fdata,dim),1,diff))!=0){
      stop("Error: objects in fdata have different dimensions")}
  } else {
    if(sum(diff(sapply(fdata, function(i) length(i))))!=0){
      stop("Error: objects in fdata have different dimensions")}}
  print("t")
  #source(paste(paste(paste("m","kernel",sep=""),"Choices",sep=""),"R",sep="."))
  checkmate::assertChoice(kernel,choices=mkernelChoices())
  checkmate::assertIntegerish(nderiv,lower=0L)
  checkmate::assertIntegerish(h,lower=0L)
  checkmate::assertLogical(parallel)
  if(!is.null(cl)){
    if(!isTRUE(parallel)){
      stop("Error: parallel should be TRUE")}
    checkmate::assertNumeric(cl,lower=1)}
  if(!is.null(measure)){
    checkmate::assertNull(metric)
    checkmate::assertChoice(measure,choices=mmeasuresChoices())
  } else {
    checkmate::assertNull(measure)
    checkmate::assertCharacter(metric)
    print(metric)
    checkmate::assertChoice(metric,choices=mmetricChoices())}
  nret=list(classes=classes,
            fdata=fdata,
            metric=metric,
            kernel=kernel,
            nderiv=nderiv,
            h=h,
            measure=measure,
            timestamps=timestamps,
            reset_timestamps=reset_timestamps,
            resample=resample,
            flip_threshold=flip_threshold,
            hover_threshold=hover_threshold,
            baseline=baseline,
            weight=weight, 
            parallel=parallel,
            cl=cl,
            diag=diag,
            upper=upper)
  class(nret)="mclassiKernel"
  return(nret)
}



# TO DO(s): 
# data transformation into funcional data
# parallelization
