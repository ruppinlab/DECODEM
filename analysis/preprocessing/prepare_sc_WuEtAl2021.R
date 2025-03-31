#### ---------------------------------------------------------------------------
#### created on 18 nov 2024, 02:58pm
#### author: dhrubas2
#### ---------------------------------------------------------------------------

setwd("/data/Lab_ruppin/projects/TME_contribution_project/analysis/analysis_final/")
source("/home/dhrubas2/vivid/miscellaneous.R")

library(Matrix)
library(reticulate)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                        # f-string print akin to python

cat("\014")                                                                     # clears console


#### ---------------------------------------------------------------------------

## get data & annotations.
data.paths  <- c("../../data/SC_data/WuEtAl2021/GSE176078_RAW/", 
                 "../../data/TransNEO/use_data/")
data.files  <- c("count_matrix_sparse.mtx", 
                 "count_matrix_genes.tsv", 
                 "count_matrix_barcodes.tsv", 
                 "metadata.csv", 
                 "gene_length_ensembl.grch37.87_SRD_17Mar2022.txt")


## get sample list for SC data.
samples.all <- list.files(data.paths[1])


## get gene annotation.
gene.annot  <- read.table(paste0(data.paths[2], data.files[5]), sep = "\t", 
                          header = T)
gene.annot  <- gene.annot %>% filter(gene_biotype == "protein_coding")


#### ---------------------------------------------------------------------------

## relevant functions.
## read per sample.
get.smpl.counts <- function(smpl){
    smpl.path <- glue("{data.paths[1]}{smpl}/")
    smpl.data <- readMM(paste0(smpl.path, data.files[1])) %>% `dimnames<-`(
        list(read.delim(paste0(smpl.path, data.files[2]), header = F)$V1, 
             read.delim(paste0(smpl.path, data.files[3]), header = F)$V1))
    smpl.data
}


## SC data is from 10X - in cellranger UMI counts format. 
## divide by total UMI counts per cell & multiply by 1M for TPM equivalence.
tpm.from.10x <- function(counts.10x){
    tpm.10x <- counts.10x %>% as.matrix %>% apply(MARGIN = 2, function(x){
        x / sum(x) * 1e6
    }) %>% as.data.frame
    tpm.10x
}


#### ---------------------------------------------------------------------------

## convert counts to TPM.
gc()                                                                            # release memory
pb      <- ProgressBar(N = samples.all %>% length)
sc.data <- samples.all %>% lapply(function(smpl){
    pb$tick()
    smpl %>% get.smpl.counts %>% tpm.from.10x
})
gc()                                                                            # release memory

sc.data <- sc.data %>% Reduce(f = cbind) %>% as.data.frame                      # build full matrix

all(abs(sc.data %>% colSums - 1e6) < 1e-4)                                      # sanity check
gc()                                                                            # release memory


## filter for protein-coding genes.
genes.pc   <- intersect(sc.data %>% rownames, gene.annot$gene_name)
sc.data.pc <- sc.data[genes.pc, ]
gc()                                                                            # release memory


#### ---------------------------------------------------------------------------

svdat <- F

if (svdat){
    out.path <- "../../data/SC_data/WuEtAl2021/"
    out.file <- c("WuEtAl2021_tpm.RDS", "WuEtAl2021_tpm_pc.RDS")
    
    fcat("saving TPM data...")
    dt <- Sys.time()
    
    saveRDS(sc.data, file = paste0(out.path, out.file[1]), compress = T)
    fcat(out.file[1])
    saveRDS(sc.data.pc, file = paste0(out.path, out.file[2]), compress = T)
    fcat(out.file[2])
    
    dt <- Sys.time() - dt
    fcat("done!");    print(dt %>% round)
}

