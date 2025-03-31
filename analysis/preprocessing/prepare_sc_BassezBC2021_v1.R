#### -------------------------------------------------------------------------------
#### created on 24 sep 2024, 01:12pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

setwd("/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/")
source("/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/r/miscellaneous.R")

library(Matrix)
library(reticulate)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## get data & annotations.
data.paths  <- c("../../data/SC_data/BassezEtAl2021/", 
                 "../../data/TransNEO/use_data/")
data.files  <- c("1863-counts_cells_cohort1.rds", 
                 "1867-counts_cells_cohort2.rds", 
                 "1872-BIOKEY_metaData_cohort1_web.csv", 
                 "1871-BIOKEY_metaData_cohort2_web.csv", 
                 "gene_length_ensembl.grch37.87_SRD_17Mar2022.txt")


bc.nac.data <- readRDS(paste0(data.paths[1], data.files[2]))
bc.nac.meta <- read.table(paste0(data.paths[1], data.files[4]), sep = ",", 
                          header = T, as.is = T, stringsAsFactors = F)


## use TransNEO annotations (hg19) to filter protein-coding genes.
gene.annot    <- read.table(paste0(data.paths[2], data.files[5]), sep = "\t", 
                            header = T, as.is = T, stringsAsFactors = F)
gene.annot.pc <- gene.annot %>% filter(gene_biotype == "protein_coding") %>% 
    (function(df) df[!duplicated(df$gene_name), ]) %>% `rownames<-`(.$gene_name)


#### -------------------------------------------------------------------------------

## get cell type counts & match names to TransNEO.
## B_cell / T_cell: B-cells / T-cells, Cancer_cell: Cancer_Epithelial, 
## Endothelial_cell: Endothelial, Fibroblast: CAFs, Mast_cell: Mast-cells, 
## Myeloid_cell: Myeloid, pDC: pDC
cell.types  <- bc.nac.meta$cellType %>% unique %>% sapply(function(ctp){
    strsplit(ctp, split = "_")[[1]][1] %>% (function(x){
        if (x %>% str_length == 1){
            paste(x, "cells", sep = "-")
        } else if (x == "Fibroblast"){
            "CAFs"
        } else if (x == "Cancer"){
            "Cancer_Epithelial"
        } else {
            x
        }
    })
}) %>% sort

cell.counts <- bc.nac.meta %>% dplyr::group_by(timepoint, cellType) %>% 
    dplyr::reframe("n_cells" = Cell %>% length) %>% as.data.frame %>% 
    `colnames<-`(c("time.point", "cell.type", "n.cells"))


#### -------------------------------------------------------------------------------

## relevant functions.
## SC data is from 10X - in cellranger UMI counts format. 
## divide by total UMI counts per cell & multiply by 1M for TPM equivalence.
tpm.from.10x <- function(counts.10x){
    tpm.10x <- counts.10x %>% as.matrix %>% apply(MARGIN = 2, function(x){
        x / sum(x) * 1e6
    }) %>% as.data.frame
    tpm.10x
}


## generate all-cell-type pseudo bulk.
## take the mean expression across all available cells for a sample.
get.pseudo.bulk <- function(sc.exp, sample.list){
    pb     <- ProgressBar(N = sample.list %>% length)
    exp.pb <- sample.list %>% sapply(simplify = F, function(smpl){
        pb$tick()
        sc.exp %>% sapply(function(sc.exp.ctp){
            cells.smpl <- sc.exp.ctp %>% colnames %>% grepl(pattern = smpl) 
            sc.exp.ctp[, cells.smpl, drop = F]
        }) %>% Reduce(f = cbind) %>% rowMeans()
    }) %>% as.data.frame
    
    exp.pb
}


#### -------------------------------------------------------------------------------

use.tp     <- "Pre"                                                                 # whether to use pre- or on-treatment samples

## build annotation matrix.
annot.dat  <- bc.nac.meta %>% `colnames<-`(c("Cell.id", "nCount_RNA", "nFeature_RNA", 
                                            "Patient.id", "time.point", "expansion", 
                                            "subtype", "cell.type", "cohort")) %>% 
    mutate("Sample.id" = paste(Patient.id, time.point, sep = "_"), 
           "cell.type" = cell.type %>% sapply(function(ctp) cell.types[ctp])) %>% 
    filter(time.point == use.tp) %>% select(
        Cell.id, Sample.id, Patient.id, cell.type, time.point, subtype, expansion, 
        nCount_RNA, nFeature_RNA)


## build cells per cell type matrix.
cell.ids   <- cell.types %>% as.character %>% sapply(simplify = F, function(ctp){
    annot.dat %>% filter(cell.type == ctp) %>% 
        select(Cell.id, Sample.id, Patient.id)
})


## build TPM expression matrices.
pb         <- ProgressBar(N = cell.types %>% length)
genes.use  <- intersect(bc.nac.data %>% rownames, gene.annot.pc$gene_name)
sc.exp.dat <- cell.types %>% as.character %>% sapply(simplify = F, function(ctp){
    pb$tick()
    cells.ctp <- cell.ids[[ctp]]$Cell.id
    exp.ctp   <- bc.nac.data[genes.use, cells.ctp] %>% tpm.from.10x
})
sc.exp.dat[["PseudoBulk"]] <- get.pseudo.bulk(
    sc.exp = sc.exp.dat, sample.list = annot.dat$Sample.id %>% unique)


## get response data.
resp.dat <- annot.dat %>% column_to_rownames("Cell.id") %>% 
    mutate("Response" = ifelse(expansion == "E", yes = 1, no = 0)) %>% 
    select(Sample.id, Patient.id, Response)


#### -------------------------------------------------------------------------------

svdat <- T

if (svdat){
    out.path <- "../../data/SC_data/BassezEtAl2021/validation/"
    dir.create(out.path, showWarnings = F)                                          # creates dir if doesn't exist already
    
    cat("saving processed data... ")
    out.file <- "bc_sc_data_bassez2021_chemo_immuno.pkl"
    out.data <- list("exp"   = sc.exp.dat %>% 
                         sapply(function(exp) exp %>% rownames_to_column("Cell.id")), 
                     "resp"  = resp.dat %>% rownames_to_column("Cell.id"), 
                     "cells" = cell.ids, 
                     "clin"  = annot.dat )
    
    py_save_object(out.data, filename = paste0(out.path, out.file), 
                   pickle = "pickle")
    
    fcat("done!\n{out.file}")
}

