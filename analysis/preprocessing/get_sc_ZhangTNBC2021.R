#### -------------------------------------------------------------------------------
#### created on 12 apr 2023, 10:58am
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(Matrix)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

data.path <- "../data/SC_data/ZhangTNBC2021/"
data.file <- "TNBC_ICB_sc_dataset.rds"

tnbc.data <- readRDS(paste0(data.path, data.file))
# tnbc.data %>% names
#   [1] "normal_count"      "raw_count"         "annotation"       
#   [4] "clinical_metadata"

tnbc.annot.all <- tnbc.data$annotation %>% as.data.frame
tnbc.clin.all  <- tnbc.data$clinical_metadata %>% as.data.frame


#### -------------------------------------------------------------------------------

## process clinical info.
# tnbc.annot.all %>% colnames
#   [1] "Cell.id"          "Sample.id"        "Patient.id"       "Origin"          
#   [5] "Tissue"           "Efficacy"         "Group"            "Treatment"       
#   [9] "Number.of.counts" "Number.of.genes"  "Cluster_org"      "Sub_cluster"     
#   [13] "Cluster_MAESTRO"

keep.cols <- c("Patient.id", "Sample.id", "Origin", "Tissue", "Group", 
               "Treatment", "Efficacy")
tnbc.patient.annot <- tnbc.annot.all[keep.cols] %>% unique %>% 
    `rownames<-`(NULL) %>% filter(Group == "Pre-treatment") %>% 
    dplyr::arrange(Treatment, Patient.id) 

tnbc.annot.pre <- tnbc.annot.all %>% `rownames<-`(NULL) %>% 
    filter(Group == "Pre-treatment")


#### -------------------------------------------------------------------------------

## filter sc data by cell type.
# tnbc.annot.all$Cluster_org %>% unique
#   [1] "B cell"       "Myeloid cell" "ILC cell"     "T cell"

cell.types <- tnbc.annot.all$Cluster_org %>% unique

pb <- ProgressBar(N = cell.types %>% length)
tnbc.exp.tissue.norm <- list();     tnbc.exp.tissue.raw <- list()
tnbc.exp.blood.norm <- list();      tnbc.exp.blood.raw <- list()
for (ctp in cell.types){
    pb$tick()
    
    ## subset tissue data.
    cells.tissue <- tnbc.annot.pre %>% 
        filter(Cluster_org == ctp, Origin == "t") %>% .$Cell.id
    
    tnbc.exp.tissue.norm[[ctp]] <- tnbc.data$normal_count[, cells.tissue]
    tnbc.exp.tissue.raw[[ctp]]  <- tnbc.data$raw_count[, cells.tissue]
    
    ## subset blood data.
    cells.blood <- tnbc.annot.pre %>% 
        filter(Cluster_org == ctp, Origin == "b") %>% .$Cell.id
    
    tnbc.exp.blood.norm[[ctp]] <- tnbc.data$normal_count[, cells.blood]
    tnbc.exp.blood.raw[[ctp]]  <- tnbc.data$raw_count[, cells.blood]
}

# tnbc.exp.tissue.raw %>% sapply(dim) %>% `rownames<-`(c("#rows", "#cols"))
#          B cell Myeloid cell ILC cell T cell
#    #rows  27085        27085    27085  27085
#    #cols  23552         9403     3304  52960
# tnbc.exp.tissue.norm %>% sapply(dim) %>% `rownames<-`(c("#rows", "#cols"))
#          B cell Myeloid cell ILC cell T cell
#    #rows  27085        27085    27085  27085
#    #cols  23552         9403     3304  52960
# tnbc.exp.blood.raw %>% sapply(dim) %>% `rownames<-`(c("#rows", "#cols"))
#          B cell Myeloid cell ILC cell T cell
#    #rows  27085        27085    27085  27085
#    #cols  17608        20737    18908  66163
# tnbc.exp.blood.norm %>% sapply(dim) %>% `rownames<-`(c("#rows", "#cols"))
#          B cell Myeloid cell ILC cell T cell
#    #rows  27085        27085    27085  27085
#    #cols  17608        20737    18908  66163


#### -------------------------------------------------------------------------------

svdat <- F                                                                          # set T to save data 
if (svdat){
    ## save sc data to use.
    out.file <- "TNBC_scExp_resp_pre_ZhangEtAl2021.RDS"
    out.obj  <- list(
        patients    = tnbc.patient.annot, 
        cells       = cell.types, 
        annot       = tnbc.annot.pre, 
        clin        = tnbc.clin.all, 
        tissue.raw  = tnbc.exp.tissue.raw, 
        tissue.norm = tnbc.exp.tissue.norm, 
        blood.raw   = tnbc.exp.blood.raw, 
        blood.norm  = tnbc.exp.blood.norm
    )
    
    saveRDS(out.obj, file = paste0(data.path, out.file), compress = T)
    
    ## save annotations.
    annot.all  <- list("annotations" = tnbc.annot.all, 
                       "clinical_metadata" = tnbc.clin.all)
    annot.file <- "TNBC_scData_ZhangEtAl2021.xlsx"
    WriteXlsx(annot.all, file.name = paste0(data.path, annot.file), row.names = F)
}









