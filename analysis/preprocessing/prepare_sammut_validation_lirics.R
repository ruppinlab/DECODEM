#### -------------------------------------------------------------------------------
#### created on 10 mar 2023, 03:31pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## read & format data.
data.path  <- "../data/TransNEO_SammutShare/out_lirics_tn_val/"
data.files <- list(ramilowski = "lirics_BRCA_Transneo-val_2015-list.rds", 
                   wang = "lirics_BRCA_Transneo-val_kun-list.rds")

cci.res <- data.files %>% sapply(simplify = F, function(file){                      # equiv to: lapply using names
    paste0(data.path, file) %>% readRDS %>% as.data.frame
})


#### -------------------------------------------------------------------------------

## save data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    out.path <- data.path
    out.file <- "lirics_BRCA_Transneo-val_SS_28Feb2023.xlsx"
    WriteXlsx(cci.res, file.name = paste0(out.path, out.file), col.names = T, 
              row.names = T, verbose = T)
}


