#### file for miscellaneous functions used often

## load often used libraries ---------------------------------------------------

library(tidyverse)
library(plyr)
library(glue)


## write functions -------------------------------------------------------------

## akin to head() for a vector.
##  x   : given matrix or dataframe
##  n   : #rows/#items (and #columns if c = NA) to display
##  c   : #columns to display. if not given, uses n
##  ord : flag to whether order the dims
Head <- function(x, n = 8, c = NA, ord = F) {
    if (is.null(dim(x))){                                                           # vector-like
        xx <- if (!ord) x else x[order(names(x))]
        nn <- if (n < length(xx)) n else length(xx)
        return( xx[1:nn] %>% `names<-`(names(x)[1:nn]) )
    }
    else {                                                                          # matrix-like
        xx <- if (!ord) x else x[order(rownames(x)), order(colnames(x))]
        rr <- if (n < nrow(xx)) n else nrow(xx)
        cc <- if (is.na(c)) n else c
        cc <- if (cc < ncol(xx)) cc else ncol(xx)
        return( xx[1:rr, 1:cc] )
    }
}


## clear console with command.
cls <- function(){ cat("\014") }


## print statements after formatting.
##  txt : strings to be appended and printed
printf <- function(..., end = "\n"){
    if (length(list(...)) > 1){ cat(sprintf(...), end) }
    else { cat(..., end) }
}


## define a progress bar.
##  N : length for progress check
##  w : width of the bar
ProgressBar <- function(N, w = 64){
    progress::progress_bar$new(format = "[:bar] :percent eta: :eta", 
                               width = w, total = N)
}


## give current date (and time) in dd-mm-yyyy (and hh-mm AM/PM) format.
##  date : return current date
##  time : return current time
DateTime <- function(date = T, time = F){
    fmt <- c("%d%b%Y", "%I%M%p")
    if (date & time){         fmt <- paste(fmt[1], fmt[2], sep = "_") }
    else if (time & !(date)){ fmt <- fmt[2] }
    else {                    fmt <- fmt[1] }
    Sys.time() %>% strftime(fmt)
}


## save multiple datasets in different sheets in a single xlsx file.
##  data.obj  : datasets saved in a named list. these names are used as sheet 
##              names by default
##  file.name : name of the xlsx file to be saved
##  sheets    : individual sheet names. if sheets = NA- uses names from data.obj 
##  row.names : flag to indicate whether to write rownames for ALL sheets 
##  col.names : flag to indicate whether to write colnames for ALL sheets 
WriteXlsx <- function(data.obj, file.name, sheets = NA, verbose = T, 
                      row.names = T, col.names = T){
    if (verbose){   pb <- ProgressBar(N = data.obj %>% length)  }
    wb <- openxlsx::createWorkbook()
    sh <- (if (is.na(sheets)) data.obj %>% names else sheets) %>% 
        `names<-`(data.obj %>% names)
    
    # write each set in a new sheet.
    data.obj %>% names %>% plyr::l_ply(function(obj){
        if (verbose){ pb$tick() }
        openxlsx::addWorksheet(wb, sh[[obj]])
        openxlsx::writeData(wb, sh[[obj]], data.obj[[obj]], rowNames = row.names, 
                            colNames = col.names)
    })
    openxlsx::saveWorkbook(wb, file = file.name, overwrite = T)
}


## evaluate prediction performance using popular metrics.
##   y_true : ground truth labels
##   y_pred : predicted labels
##   metric : a single metric or list of metrics. available metrics are 
##            "MSE"/"RMSE"/"NRMSE", "MAE"/"NMAE", "PCC"/"SCC", and "R2"
    
performance <- function(y_true, y_pred, metrics = c("NRMSE", "PCC")){
    y_true <- y_true %>% as.numeric;    y_pred <- y_pred %>% as.numeric

    # base functions.
    MAE <- function(y_t, y_p = NULL){ mean(abs( y_t - (if (is.null(y_p)) mean(y_t) else y_p) )) }
    MSE <- function(y_t, y_p = NULL){ mean(( y_t - (if (is.null(y_p)) mean(y_t) else y_p) )^2) }
    
    # calculate performance.
    perf = c()
    for (met in metrics){
        met <- met %>% toupper
        if (grepl(met, pattern = "MSE")){                                       # squared errors
            res <- MSE(y_true, y_pred)
            if (met == "NRMSE"){        res <- sqrt(res / MSE(y_true))  }
            else if (met == "RMSE"){    res <- sqrt(res)    }
        }
        else if (grepl(met, pattern = "MAE")){                                  # absolute errors
            res <- MAE(y_true, y_pred)
            if (met == "NMAE"){         res <- res / MAE(y_true)    }
        }
        else if (grepl(met, pattern = "CC")){                                   # correlations
            res <- cor(y_true, y_pred, method = (if (met == "SCC") "spearman" else "pearson"))
        }
        else if (met == "R2"){                                                  # r-squared
            res <- summary(lm(y_pred ~ y_true))$r.squared
            # res <- cor(y_true, y_pred, method = "pearson")^2
            #
            # res <- 1 - MSE(y_true, y_pred) / MSE(y_true)
            # if (res < 0){     res <- 0    }
        }
        else {
            stop("Unknown metric!")
        }
            
        perf[met] <- res                                                        # save value
    }
    perf
}

