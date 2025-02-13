#'mmeasures: computes measures from mousetrap package
#'
#'mtdata is a data frame with id, x- and y-coordinates (in this order)
#'timestamps is a data frame with the timestamps of the trajectories (training + testing order)
#'reset_timestamps is a logical value from mt_import_mousetrap (see moustrap package). Defaul FALSE
#'reample is a logical value. When TRUE (default) the function mt_resample is executed (see moustrap package)
#'flip_threshold is a numerical value from mt_measures (see moustrap package)
#'hover_threshold is a numerical value from mt_measures (see moustrap package)
#'@export
mmeasures=function( mtdata,
                    timestamps,
                    reset_timestamps=FALSE,
                    resample=TRUE,
                    flip_threshold=0,
                    hover_threshold=2000)
{
  options(warn=-1)
  checkmate::assertList(mtdata)
  checkmate::assertList(timestamps)
  checkmate::assertLogical(reset_timestamps)
  checkmate::assertLogical(resample)
  checkmate::assertNumeric(flip_threshold ,lower=0,len=1L)
  checkmate::assertNumeric(hover_threshold,lower=0,len=1L)
  
#'@export
  mousetrapDataFormat=function(data)
  {
    checkmate::assertDataFrame(data)
    return(data.frame(xpos=as.factor(paste(paste("[",paste(data$xpos,collapse=","),sep=""),"]",sep="")),ypos=as.factor(paste(paste("[",paste(data$ypos,collapse=","),sep=""),"]",sep="")),timestamps=as.factor(paste(paste("[",paste(data$timestamps,collapse=","),sep=""),"]",sep=""))))}

  mtdata=lapply(plyr::alply(cbind(1:ncol(mtdata[[1]])),1,function(i) lapply(mtdata,function(j) cbind(j[,i]))),function(x) do.call(cbind,x))
  mtdata=do.call(rbind,lapply(plyr::alply(cbind(1:length(mtdata)),1,function(i) data.frame(xpos=mtdata[[i]][,1],ypos=mtdata[[i]][,2],timestamps=timestamps[[i]])),mousetrapDataFormat))
  mtdata=cbind(identifier=1:nrow(mtdata),mtdata)
  mtdata=mousetrap::mt_import_mousetrap(mtdata,reset_timestamps=reset_timestamps,mt_id_label="identifier")
  if(isTRUE(resample)) mtdata=mousetrap::mt_resample(mtdata,use="trajectories",c("xpos","ypos"),timestamps="timestamps",save_as="trajectories")
  mtmeasures=mousetrap::mt_measures(mtdata,use="trajectories",flip_threshold=flip_threshold,hover_threshold=hover_threshold)$measures
  return(mtmeasures)
}
