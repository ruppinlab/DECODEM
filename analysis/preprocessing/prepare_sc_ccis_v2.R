#### -------------------------------------------------------------------------------
#### created on 14 aug 2023, 03:52pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

read.data <- function(file, sheet = 1, sep = "\t", header = T, index = F){
    if (file %>% strsplit(split = ".", fixed = T) %>% .[[1]] %>% 
        (function(x) x[length(x)]) == "xlsx"){
        data <- openxlsx::read.xlsx(file, sheet = sheet, colNames = header, 
                                    rowNames = index, check.names = F)
    } else {
        indexx <- if(index) 1 else NULL
        data   <- read.table(file, sep = sep, header = header, row.names = indexx, 
                             check.names = F, stringsAsFactors = F)
    }
}


fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## get data & annotations.
data.paths <- c("../data/TransNEO/transneo_analysis/mdl_data/", 
                "../data/SC_data/ZhangTNBC2021/")

data.files <- c(
    "brightness_lirics_feature_importance_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_v2_14Aug2023.xlsx", 
    "brightness_lirics_feature_list_chemo_RF_allfeatures_3foldCV_v2_14Aug2023.xlsx", 
    "TNBC_scExp_tissue_pre_Chemo_ZhangEtAl2021_B-cells.tsv", 
    "TNBC_scExp_tissue_pre_Chemo_ZhangEtAl2021_Myeloid.tsv", 
    "TNBC_scExp_tissue_pre_Chemo_ZhangEtAl2021_ILC.tsv", 
    "TNBC_scExp_tissue_pre_Chemo_ZhangEtAl2021_T-cells.tsv", 
    "TNBC_scAnnot_tissue_pre_ZhangEtAl2021.xlsx"
)


#### -------------------------------------------------------------------------------

## list of top CCIs.
cci.bn     <- read.data(paste0(data.paths[1], data.files[1]))
cci.gp     <- cci.bn %>% select(LigandGene, ReceptorGene) %>% 
    (function(df) df[!duplicated(df), ])
cci.genes  <- union(cci.bn$LigandGene, cci.bn$ReceptorGene)

cci.all.bn <- read.data(paste0(data.paths[1], data.files[2]), sheet = "ramilowski")
colnames(cci.all.bn)[1] <- "CCIannot"
cci.gp.all <- cci.all.bn %>% select(LigandGene, ReceptorGene) %>% 
    (function(df) df[!duplicated(df), ])
cci.genes.all <- union(cci.all.bn$LigandGene, cci.all.bn$ReceptorGene)


#### -------------------------------------------------------------------------------

## prepare SC data.
sc.bcells  <- read.data(paste0(data.paths[2], data.files[3]), index = T)
sc.myeloid <- read.data(paste0(data.paths[2], data.files[4]), index = T)
sc.ilc     <- read.data(paste0(data.paths[2], data.files[5]), index = T)
sc.tcells  <- read.data(paste0(data.paths[2], data.files[6]), index = T)

# sc.exp.bn  <- cbind(sc.bcells[cci.genes, ], sc.myeloid[cci.genes, ],
#                     sc.ilc[cci.genes, ], sc.tcells[cci.genes, ]) %>% as.data.frame

cci.genes     <- intersect(cci.genes, sc.bcells %>% rownames)
sc.exp.bn     <- cbind(sc.bcells[cci.genes, ], sc.myeloid[cci.genes, ], 
                       sc.tcells[cci.genes, ]) %>% as.data.frame

remove.gene   <- setdiff(cci.genes.all, sc.bcells %>% rownames)
cci.genes.all <- setdiff(cci.genes.all, remove.gene)
cci.gp.all    <- cci.gp.all %>% filter(
    ((LigandGene %in% remove.gene) | (ReceptorGene %in% remove.gene)) %>% `!`)
sc.exp.all.bn <- cbind(sc.bcells[cci.genes.all, ], sc.myeloid[cci.genes.all, ], 
                       sc.tcells[cci.genes.all, ]) %>% as.data.frame


## prepare SC annotations.
keep.cols <- c("Cell.id", "Sample.id", "Tissue", "Number.of.counts", 
               "Number.of.genes", "Cell.type", "Response")
sc.annot  <- read.data(paste0(data.paths[2], data.files[7]), sheet = "Chemo")
sc.pheno  <- sc.annot %>% column_to_rownames("Cell.id") %>% 
    (function(df) df[sc.exp.bn %>% colnames, ]) %>% 
    mutate(Response = (Efficacy == "PR") %>% as.integer) %>%                        # CR / PR := R, only PR is available in data
    rownames_to_column("Cell.id") %>% select(all_of(keep.cols))


#### -------------------------------------------------------------------------------

## make combined object.
svdat <- F                                                                          # set T to save data 
if (svdat){
    
    datestamp <- DateTime()
    out.path  <- data.paths[1]
    out.file  <- glue::glue("SC_data_for_top_{data.obj$n.cci}CCIs_SRD_{datestamp}.RDS")
    data.obj <- list("n.cci" = cci.bn %>% nrow, "cci.list" = cci.bn, 
                     "cci.list.gp" = cci.gp, "cci.genes" = cci.genes, 
                     "n.cci.all" = cci.all.bn %>% nrow, 
                     "cci.list.all" = cci.all.bn, "cci.list.gp.all" = cci.gp.all, 
                     "cci.genes.all" = cci.genes.all, "sc.exp" = sc.exp.bn, 
                     "sc.exp.all" = sc.exp.all.bn, "sc.annot" = sc.pheno)
    
    saveRDS(data.obj, file = paste0(out.path, out.file), compress = T)
}


## truncated data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    datestamp <- DateTime()
    out.path  <- data.paths[1]
    out.file  <- glue::glue("SC_data_for_Sahil_{cci.all.bn %>% nrow}CCIs_{cci.gp.all %>% nrow}genepairs_SRD_{datestamp}.RDS")
    data.obj.trn <- list("cci.list.gp.all" = cci.gp.all, 
                         "sc.exp.all" = sc.exp.all.bn, 
                         "sc.annot" = sc.pheno)

    saveRDS(data.obj.trn, file = paste0(out.path, out.file), compress = T)
}


