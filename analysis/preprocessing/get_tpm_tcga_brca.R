#### -------------------------------------------------------------------------------
#### created on 06 apr 2023, 06:05pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## get TCGA data for all samples/genes.
data.path <- c("../data/TCGA/", 
               "../data/TransNEO/")
data.file <- c("TCGA_RSEM_gene_symbol_tpm.txt", 
               "tcga_survival.tsv", 
               "gene_length_ensembl.grch37.87_SRD_17Mar2022.txt")

tcga.tpm  <- data.table::fread(paste0(data.path, data.file[1])) %>% as.data.frame
tcga.surv <- data.table::fread(paste0(data.path, data.file[2])) %>% as.data.frame


## remove duplicated genes.
genes.dup <- tcga.tpm$Gene %>% table %>% 
    (function(freq) freq[freq > 1] %>% names) %>% 
    sapply(function(gn) which(tcga.tpm$Gene == gn)) %>% Reduce(f = union)

tcga.tpm.nodup <- tcga.tpm[-genes.dup, ] %>% `rownames<-`(NULL) %>% 
    column_to_rownames("Gene")


#### -------------------------------------------------------------------------------

## filter samples for BRCA.
samples.tpm  <- tcga.tpm.nodup %>% colnames %>% sapply(function(smpl){
    strsplit(smpl, split = "-")[[1]][1:3] %>% paste(collapse = "-")
})

samples.brca <- tcga.surv[tcga.surv$cancer == "BRCA", "tcga_id"] %>% 
    intersect(., samples.tpm)

tcga.tpm.brca <- tcga.tpm.nodup %>% `colnames<-`(samples.tpm) %>% 
    .[, samples.brca]


## filter for protein-coding genes.
gene.annot.tn <- data.table::fread(paste0(data.path[2], data.file[3])) %>% as.data.frame

genes.pc <- gene.annot.tn[
    gene.annot.tn$gene_biotype == "protein_coding", "gene_name"] %>% 
    intersect(., tcga.tpm.brca %>% rownames)

tcga.tpm.brca.pc <- tcga.tpm.brca[genes.pc, ]


## data is in log-scale- so transform back to TPM.
tcga.tpm.brca.pc.logrev <- (2 ^ tcga.tpm.brca.pc) %>% as.data.frame
tcga.tpm.brca.pc.logrev <- tcga.tpm.brca.pc.logrev %>% apply(MARGIN = 1, var) %>% 
    (function(gn.var) which(gn.var > 0)) %>% tcga.tpm.brca.pc.logrev[., ]           # filter genes with var = 0


#### -------------------------------------------------------------------------------

svdat <- F                                                                          # set T to save data 
if (svdat){
    out.file <- data.file[1] %>% gsub(pattern = ".txt", replacement = "_pc_BRCA.txt")
    write.table(tcga.tpm.brca.pc.logrev %>% rownames_to_column("Gene"), 
                file = paste0(data.path, out.file), sep = "\t", col.names = T, 
                row.names = F)
}



