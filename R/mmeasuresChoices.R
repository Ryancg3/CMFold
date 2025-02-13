#' mmeasuresChoices
#' @export

mmeasuresChoices=function(proxy.only=FALSE)
{
  proxy_measures_names=c("xpos_max",
			 "xpos_min",
			 "ypos_max",
			 "ypos_min",
			 "MAD",
			 "MAD_time",
			 "MD_above",
			 "MD_above_time",
			 "MD_below",
			 "MD_below_time",
			 "AD",
			 "AUC",
			 "xpos_flips",
			 "ypos_flips",
			 "flips",
			 "xpos_reversals",
			 "ypos_reversals",
			 "RT",
			 "initiation_time",
			 "idle_time",
			 "hover_time",
			 "hovers")
  return(proxy_measures_names)
}

