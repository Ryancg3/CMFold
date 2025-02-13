#' Sequence alignment scores between multivariate functions / trajectories
#'
#' Computes Sequence Alignment scores for all pairs of \eqn{m}-dimensional functions.
#' Sequence alignment scores are usually computed for aligning sequences of nominal values. The present function extends 
#' @param dim_min_max list of size \eqn{m} (number of dimensions), containing vectors of size 2 giving the minimum and maximum values of the functions in each dimensions.
#' @param n_breaks vector of integers giving the number of breaks in each dimension. Limited to max 10 per dimension and max 500 areas in total.
#' @param custom_splits list of size \eqn{m} (number of dimensions of the functions) containing vectors giving the exact value at which to include splits in each dimension. In the format list(c(lower bound, break1, break2,..., breakn, upperbound), ...). Either define \code{custom_splits} or (\code{dim_min_max} and \code{n_breaks})


malignmentscore = function(data,#This function takes as input a list of data.frames (one data frame for each dimension)
                                                  #Columns contain data for different Individual 
                                                   #Rows contain data for different time stamps
                                       Keeprep= TRUE , # should consecutive locations in the same area be kept? or dropped?
                                       dim_min_max = NULL, # list of value range in each dimension 
                                                          # IMPORTANT: all locations should be contained in the specified range , otherwise NA's are returned
                                       #in each dimension give c(lowest possible value, highest possible value)
                                       n_breaks = NULL,
                                       custom_splits = NULL, # list of vectors for custom area divisions in the format list(c(lower bound, break1, break2,..., breakn, upperbound))
                                       # either custom_splits or (dim_min_max and n_breaks )
                                       ... #options for the Biostrings::pairwise alignment function
                                       ){
  ###################################################################################
  ##1. TESTS FOR CORRECT INPUT
  ###################################################################################
  if(is.null(custom_splits)==FALSE & (is.null(dim_min_max)==FALSE | is.null(n_breaks)==FALSE) ){
    stop("Error: Either 'custom_splits' have to be specified, or parameters for creating equally spaced splits '' ''")
    }
  
  if (is.null(n_breaks)==FALSE){
    checkmate::assert_vector(n_breaks)
    if(length(n_breaks)!= length(data)){
      stop("Error: n_breaks and data do not have same number of dimensions")
    }
    if (max(n_breaks)>10){
      stop("Only 10 breaks per dimension are allowed")
    }
    
    if (prod(n_breaks)>500){
      stop("Only up to 500 Areas splits are allowed. Reduce the number of breaks in n_breaks")
    }
  }
  
  if (is.null(custom_splits)==FALSE){
    checkmate::assert_list(custom_splits)
    if(length(custom_splits)!= length(data)){
      stop("Error: custom_splits and data do not have same number of dimensions")
    }
  }
  
  if (is.null(dim_min_max)==FALSE){
    checkmate::assert_list(dim_min_max)
    if(length(dim_min_max)!= length(data)){
      stop("Error: dim_min_max and data do not have same number of dimensions")
    }
  }
  
  checkmate::assertLogical(Keeprep)

  ##################################################################################
  ## 2. Preprocess data: transform coordinates to "areas"
  ##################################################################################
  
  ##2.1 Create break list or take custom from user
  
  if(is.null(custom_splits)){
    
    #Create 1D break list
    Breaks_1D = function(dim_min_max, # vector c(min_value, max_value))
                         n_breaks) {  #number of breaks 
      #returns vector of breaks according to min max values and number of breaks
      
      return(seq(from= dim_min_max[1]-abs(dim_min_max[1])/1000000, to= dim_min_max[2]+dim_min_max[2]/1000000, length.out = n_breaks+1 ))
    }
    
    #Create ND break list
    Breaks_ND = mapply(Breaks_1D, 
                       dim_min_max,
                       n_breaks,
                       SIMPLIFY =  FALSE
    )
  }else{
    Breaks_ND = custom_splits
    n_breaks = sapply(Breaks_ND, function(x) length(x)-1)
    }

  ##2.2 categorize data into breaks 
  
  #categorize data into Breaks (1D)
  categorize_1D = function(x, Breaks){
    apply(x, #functiunnal data in 1 D
          2, #apply on column
          function(x) cut(x, breaks=Breaks, labels=0:(length(Breaks)-2)) #take values of x in categorize them according to breaks
    )
  }
  
  #mapply categorize 1D in all dimensions 
  categorize_ND = function(x, Breaks_ND){
    mapply(categorize_1D, #apply categorize 1D om every dimension and every break list
           data,
           Breaks_ND, 
           SIMPLIFY = FALSE)
  }
  
  #return for each data point the area in each dimension (e.g area 1 in dimension1, area 3 in dimension2 ...)
  x = categorize_ND(x, #input data
                    Breaks_ND) # breaks previsouly calculated
  
  # convert list into 3d array
  x = array(unlist(x), dim=c(nrow(x[[1]]),NCOL(x[[1]]),length(x))) 
  
  #2.3 Create "overall" areas, in the sense the sense that Area_D1 = 1, Area_D2 = 3 --> result = Area "12"
  #concatenate 1 D dimension into ND. e.g: if 1st Dim= Area "1", 2nd Dim=Area "2" then result = area "12"
  x = apply(x,
            c(1,2),
            function(x) paste(x, collapse  =""))

  ##2.4 Translate sequences of areas 112, 113, 114 ... into character strings "eae" for package biostring
  #map sequence of area ID's to characters
  
  Area_Codes = lapply(as.list(n_breaks),
                     function(x) return(0:(x-1)))
  
  Area_EncodedasNumbers = apply(expand.grid(Area_Codes), 
                                1,
                                function(x) paste(x, collapse  =""))
  
  #transform area codes to characters (e.g 111 --> "e")
  Allowed_characters = c(intToUtf8(33:126, multiple=TRUE), intToUtf8(161:540, multiple=TRUE))
  Area_Encodedascharacter = Allowed_characters[rank(as.numeric(Area_EncodedasNumbers))]
  
  #map area codes to characters in the data
  x = as.list(data.frame(mapvalues(x,
                                   from = Area_EncodedasNumbers ,
                                   to= Area_Encodedascharacter,
                                   warn_missing = FALSE)
                         ,stringsAsFactors = FALSE))
  
  #delete adjacent duplicates aaabbbaaaccc --> abac (optional)
  if (Keeprep == FALSE){x = lapply(x, function(y) rle(y)[[2]])}
  
  # form string of characters for each individual (position "DNA")
  x= sapply(x, function(x) paste(x, collapse = ""), simplify= TRUE )

  #################################################################################
  #3 Implement Pairwise alignment criterias
  #################################################################################
  
  #pairwise alignment with repeating areas
  pairwise_mat = Vectorize(function(i,j,x){
    return(pairwiseAlignment(x[i],x[j], scoreOnly=TRUE, ...))
  },vectorize.args=c("i","j") )
  
  output = outer(1:length(x),
                 1:length(x),
                 FUN = pairwise_mat,
                 x)
  return(output)
  }
