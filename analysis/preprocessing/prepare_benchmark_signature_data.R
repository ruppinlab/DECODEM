#### ---------------------------------------------------------------------------
#### created on 22 nov 2024, 12:57pm
#### author: dhrubas2
#### ---------------------------------------------------------------------------

setwd("/data/Lab_ruppin/projects/TME_contribution_project/analysis/analysis_final/")
source("/home/dhrubas2/vivid/miscellaneous.R")

library(Seurat)
library(presto)
library(Matrix)
library(reticulate)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                        # f-string print akin to python

cat("\014")                                                                     # clears console


#### ---------------------------------------------------------------------------

## ======= Prepare signature files =======
# outdir="../results/ICB_deconvolution/inputs"
outdir <- "../../data/SC_data/WuEtAl2021/"
if (!dir.exists(outdir)){
    dir.create(outdir, recursive = T)
}

## read data for signature generation.
sc_tpm  <- readRDS("../../data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_sc_tpm_for_signature.RDS")
sc_meta <- readRDS("../../data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_sc_meta_for_signature.RDS")
all(sc_meta %>% rownames == sc_tpm %>% colnames)                                # sanity check


## ------- Prepare data for CIBERSORTX  -------
## FULL DATA
sig_cpm <- sc_tpm %>% rownames_to_column(var = "GENE") %>% as.matrix
colnames(sig_cpm)[2:ncol(sig_cpm)] <- sc_meta$celltype

write.table(sig_cpm, file = paste0(outdir, "WuEtAl2021_benchmark_signature_data.txt"), 
            sep = "\t", quote = F, row.names = F, col.names = T)


## DOWNSAMPLING
# tmp_meta=sc_obj_filter@meta.data
# tmp_meta$cell_ids=rownames(tmp_meta)
tmp_meta <- sc_meta

# tmp_meta_v2=tmp_meta[,c("cell_ids","revised_major_state","sample")]
tmp_meta_v2 <- tmp_meta[c("cell", "celltype", "patient")]
colnames(tmp_meta_v2) <- c("cell_ids","cell_types","sample_ids")

# cpm_norm_count=sc_obj_filter@assays$RNA$data
cpm_norm_count <- sc_tpm

## down sampling 
pct <- 0.075
sub_norm_count <- NULL
sub_cell_ids <- NULL
for (i in unique(tmp_meta$celltype_major)){
    print(i)
    tmp_df <- tmp_meta[which(tmp_meta$celltype_major == i), ]
    set.seed(7)
    tmp_id <- sample(rownames(tmp_df), replace = F, size = round(nrow(tmp_df) * pct))
    print(length(tmp_id))
    sub_cell_ids <- c(sub_cell_ids, tmp_id)
    tmp_count <- cpm_norm_count[, tmp_id]
    
    tmp_count2 <- as.data.frame(t(data.frame(cell_type <- rep(i, ncol(tmp_count)))))
    colnames(tmp_count2) <- tmp_id
    tmp_count2 <- rbind(tmp_count2, as.data.frame(tmp_count))
    
    if(is.null(sub_norm_count)){
        sub_norm_count <- tmp_count2
    } else{
        sub_norm_count <- cbind(sub_norm_count, tmp_count2)
    }
}

# Remove rows with all zeros value
sub_norm_count_filtered <- sub_norm_count[rowSums(sub_norm_count != 0) > 0, ]
rownames(sub_norm_count_filtered)[1] <- "GeneSymbols"
# sub_norm_count_filtered <- sub_norm_count_filtered %>% rownames_to_column("GeneSymbols")
# sub_norm_count_filtered[1, 1] <- "GENE"

dim(sub_norm_count_filtered)

# write.table(sub_norm_count_filtered, file.path(outdir, "bill_downsampling_norm_count.txt"), quote = F, sep = "\t", col.names = F)
write.table(sub_norm_count_filtered, file.path(outdir, "WuEtAl2021_benchmark_signature_data_downsampled.txt"), 
            quote = F, sep = "\t", col.names = F, row.names = F)

