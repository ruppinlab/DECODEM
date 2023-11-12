#### -------------------------------------------------------------------------------
#### created on 16 feb 2023, 01:04pm
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
data.path  <- "../data/TransNEO/LIRICS_results/for_Dhruba_Jan19/"
data.files <- list(ramilowski = "lirics_BRCA_Transneo_2015-list.Rdata", 
                   wang = "lirics_BRCA_Transneo_kun-list.Rdata")

cci.res <- data.files %>% sapply(simplify = F, function(file){                      # equiv to: lapply using names
    load(paste0(data.path, file))                                                   # loads an object named "lirics_output"
    lirics_output %>% as.data.frame
})


#### -------------------------------------------------------------------------------

## save data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    out.path <- paste0(data.path, "../")
    out.file <- "lirics_BRCA_Transneo_SS_19Jan2023.xlsx"
    WriteXlsx(cci.res, file.name = paste0(out.path, out.file), col.names = T, 
              row.names = T, verbose = T)
}


