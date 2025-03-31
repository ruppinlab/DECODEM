#### ---------------------------------------------------------------------------
#### created on 19 nov 2024, 11:25pm
#### author: dhrubas2
#### ---------------------------------------------------------------------------

setwd("/data/Lab_ruppin/projects/TME_contribution_project/analysis/analysis_final/")
source("/home/dhrubas2/vivid/miscellaneous.R")

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                        # f-string print akin to python

cat("\014")                                                                     # clears console


#### ---------------------------------------------------------------------------

## helper function to generate technical replicates by injecting mRNA composition noise:
# n_genes: number of genes to randomly pick and add noise to 
# amp_factor: scalar to increase or decrease the TPM value of a gene, simulating a 
#             PCR amplification bias or degradation of mRNA in a preserved sample
add_noise <- function(bulk, n_genes = 1000, amp_factor){
    bulk_new <- apply(bulk, MARGIN = 2, function(x){
        idx <- sample(seq(nrow(bulk)), n_genes)
        xx  <- x
        xx[idx] <- x[idx] * amp_factor
        # renormalizing
        xx <- xx / sum(xx) * 1e6
        return(xx)
    })
    return(bulk_new)
}

# library(readr)
# library(dplyr)
# library(ggplot2)

#### ---------------------------------------------------------------------------
## read in melanoma single cell dataset
# format: sc = expression matrix with cols as cell ids and rows as genes
#         meta = metadata table with rows as cell ids (matched in order to 
#                cols of sc matrix), a cell_type column with celltype labels 
#                for each cell & a patient column with the label of the patient 
#                sample from which the cell comes

# NOTE: the file paths here are local. Please specify your own file path to the 
# single cell RNASeq data and meta data for reading
sc <- readRDS("/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_tpm_pc.RDS")
# sc <- read_delim(file = "~/forKun/Melanoma_scRNASeq/scimpute_tpm.txt",delim = "\t")
# sc <- as.data.frame(sc)
# rownames(sc) <- sc$X1
# sc <- sc[,-1]

# meta <- readRDS("~/forKun/Melanoma_scRNASeq/Livnat_updated_celltypes_table.rds")
# meta$celltype <- as.character(meta$celltype)
# meta$cell <- as.character(meta$cell)
# meta$patient <- as.character(meta$patient)
meta <- read.table("/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/Wu_etal_2021_BRCA_scRNASeq/metadata.csv", 
                   sep = ",", header = T, row.names = 1)
meta <- meta[sc %>% colnames, ]
meta$celltype <- meta$celltype_major
meta$cell <- meta %>% rownames %>% 
    sapply(function(x) strsplit(x, split = "_")[[1]][2])
meta$patient <- meta$orig.ident

## set sc data of 4 patients aside to derive cell-type signatures/markers.
k <- 4
set.seed(89128)
idx <- as.character(meta$patient) %in% sample(unique(as.character(meta$patient)), k)
meta_patient <- meta %>% group_by(patient) %>% dplyr::count(celltype)
print(table(meta$celltype[idx]))

sc_signature   <- sc[, idx]
meta_signature <- meta[idx, ]

## generate pseudo bulk and pure cell type specific expression profiles of 
## remaining patients by averaging single cell data cell types and patients
# cctypes <- c("Mal","Endo.","CAF","T.CD8","NK","Macrophage","pDC","skinDC","T.CD4","B.cell")
cctypes  <- meta$celltype %>% unique %>% sort
bm_celltypes <- as.character(meta$celltype)
patients <- as.character(meta$patient)

bm_cell_fracs <- sapply(unique(patients[!idx]), function(x) {
    counts <- sapply(cctypes, function(y) {
        return(sum(patients %in% x & bm_celltypes == y))
    })
    counts <- counts / sum(counts)
    names(counts) <- cctypes
    return(counts)
})
colnames(bm_cell_fracs) <- unique(patients[!idx])

## generate ground truth cell-type-specific expression
pb <- ProgressBar(N = patients[!idx] %>% unique %>% length)
bm_ct_expr <- lapply(unique(patients[!idx]), function(x) {
    pb$tick()
    cdat <- sapply(cctypes, function(y) {
        if(sum(patients %in% x & bm_celltypes == y) > 1){
            return(rowMeans(sc[, patients %in% x & bm_celltypes == y], na.rm = T))
        }
        else if(sum(patients %in% x & bm_celltypes == y) == 1){
            return(sc[, patients %in% x & bm_celltypes == y])
        }
        else{
            return(rep(0, nrow(sc)))
        }
    })
})
names(bm_ct_expr) <- unique(patients[!idx])

## transpose the cell-type specific expression profiles for each patient
bm_ct_expr2 <- lapply(cctypes, function(x){
    return(sapply(names(bm_ct_expr), function(y){
        return(bm_ct_expr[[y]][, x])
    }))
})
names(bm_ct_expr2) <- cctypes

## generate expected bulk
bm_bulk <- sapply(names(bm_ct_expr), function(x) {
    bb <- bm_ct_expr[[x]] %*% bm_cell_fracs[, x]
})
rownames(bm_bulk) <- rownames(sc)
bm_cell_fracs <- t(bm_cell_fracs)


## generate technical replicates with batch effects
bm_bulk_noisy2 <- add_noise(bm_bulk, n_genes = 3000, amp_factor = runif(3000, max = 50))
bm_bulk_noisy  <- add_noise(bm_bulk, n_genes = 500, amp_factor = runif(500, min = 10, max = 50))
labs     <- c(rep("bulk_noise_free", bm_bulk %>% ncol), rep("bulk_noisy", bm_bulk %>% ncol), 
              rep("bulk_noisy2", bm_bulk %>% ncol))
combined <- apply(cbind(bm_bulk, bm_bulk_noisy, bm_bulk_noisy2), MARGIN = 2, scale)
pca1     <- prcomp(t(combined))

# sanity check: if mRNA noise addition introduces systematic batch effects
p1 <- ggplot(data = data.frame(dim1 = pca1$x[, 1], dim2 = pca1$x[, 2], dataset = labs, stringsAsFactors = F), 
             mapping = aes(x = dim1, y = dim2, color = labs)) + geom_point() + theme_minimal()
print(p1)


# ############################################### SAVE RESULTS #######################################################
# ground truth cell fractions for 10 cell types in each patient
# saveRDS(Livnat_cell_fracs, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_cellfracs_no_mix_CV.rds")
saveRDS(bm_cell_fracs, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_cell_fraction_no_mix.RDS")

# ground truth cell type specific expression in each patient: used for CODEFACS and CIBERSORTx performance evaluation
# saveRDS(Livnat_ct_expr2, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_groundtruth_tpm_no_mix_CV.rds")
saveRDS(bm_ct_expr2, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_cell_type_tpm_no_mix.RDS")

# pseudo-bulk gene expression for each of the patients
# saveRDS(livnat_bulk, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_bulk_tpm_no_mix_CV.rds")
# saveRDS(bm_bulk, file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_no_mix.RDS")

# pseudo-bulk gene expression for each of the technical replicates
# saveRDS(livnat_bulk_noisy, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_bulk_tpm_no_mix_noisy_CV.rds")
# saveRDS(livnat_bulk_noisy2, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_bulk_tpm_no_mix_noisy2_CV.rds")
# saveRDS(bm_bulk_noisy, file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_no_mix_noisy.RDS")
# saveRDS(bm_bulk_noisy2, file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_no_mix_noisy2.RDS")

# scRNASeq matrix for generation of signature (apply any publicly available signature generation tool 
# on this data. We used CIBERSORTx signature generation module for our benchmarking and subsequent 
# analyses as they yield the best quality signatures)
# saveRDS(sc_signature, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_forSignature_CV.rds")
saveRDS(sc_signature, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_sc_tpm_for_signature.RDS")

# metadata for generation of signature
# saveRDS(meta_signature, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_meta_forSignature_CV.rds")
saveRDS(meta_signature, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_sc_meta_for_signature.RDS")

# saving bulk mattrices
# write.table(livnat_bulk, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_no_mix_CV.txt",sep = "\t", quote = F, col.names = NA)
# write.table(livnat_bulk_noisy, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_no_mix_noisy_CV.txt",sep = "\t", quote = F, col.names = NA)
# write.table(livnat_bulk_noisy2, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_no_mix_noisy2_CV.txt",sep = "\t", quote = F, col.names = NA)
write.table(bm_bulk %>% as.data.frame %>% rownames_to_column(var = "GENE"), 
            file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_no_mix.tsv", 
            sep = "\t", quote = F, row.names = F, col.names = T)
write.table(bm_bulk_noisy %>% as.data.frame %>% rownames_to_column(var = "GENE"), 
            file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_no_mix_noisy.tsv", 
            sep = "\t", quote = F, row.names = F, col.names = T)
write.table(bm_bulk_noisy2 %>% as.data.frame %>% rownames_to_column(var = "GENE"), 
            file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_no_mix_noisy2.tsv", 
            sep = "\t", quote = F, row.names = F, col.names = T)

# saving gene names
write.table(bm_bulk %>% rownames, 
            file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_genes_all.tsv", 
            sep = "\t", quote = F, row.names = F, col.names = F)
# ############################################### SAVE RESULTS #######################################################


################################################# mixing experiment 1 ################################################
## 100 pseudobulk sample where cells from any cell type can be randomly taken 
## from more than one patient to generate a mixture

set.seed(89128)
## cell types and patients
# cctypes = c("Mal","Endo.","CAF","T.CD8","NK","Macrophage","pDC","skinDC","T.CD4","B.cell")
cctypes  <- meta$celltype %>% unique %>% sort
bm_mix_celltypes <- as.character(meta$celltype)
patients <- as.character(meta$patient)
combs <- t(combn(unique(patients[!idx]), 4))
mixes <- sample(seq(nrow(combs)), 100)
print(mixes)

## generate cell fractions
bm_mix_cell_fracs <- sapply(mixes, function(x) {
    counts <- sapply(cctypes, function(y) {
        return(sum(patients %in% combs[x, ] & bm_mix_celltypes == y))
    })
    counts <- counts / sum(counts)
    names(counts) <- cctypes
    return(counts)
})
colnames(bm_mix_cell_fracs) <- sapply(mixes, function(x) paste("mix", x, sep = ""))

## generate ground truth
pb <- ProgressBar(N = mixes %>% length)
bm_mix_ct_expr <- lapply(mixes, function(x) {
    pb$tick()
    cdat <- sapply(cctypes, function(y) {
        if(sum(patients %in% combs[x, ] & bm_mix_celltypes == y) > 1){
            return(rowMeans(sc[, patients %in% combs[x, ] & bm_mix_celltypes == y], na.rm = T))
        }
        else if(sum(patients %in% combs[x, ] & bm_mix_celltypes == y) == 1){
            return(sc[, patients %in% combs[x, ] & bm_mix_celltypes == y])
        }
        else{
            return(rep(0, nrow(sc)))
        }
    })
})
names(bm_mix_ct_expr) <- sapply(mixes, function(x) paste("mix", x, sep = ""))

## transposing expression profiles
bm_mix_ct_expr2 <- lapply(cctypes, function(x){
    return(sapply(names(bm_mix_ct_expr), function(y){
        return(bm_mix_ct_expr[[y]][, x])
    }))
})
names(bm_mix_ct_expr2) = cctypes

## generate expected bulk
bm_mix_bulk <- sapply(names(bm_mix_ct_expr), function(x) {
    bb <- bm_mix_ct_expr[[x]] %*% bm_mix_cell_fracs[, x]
})
rownames(bm_mix_bulk) = rownames(sc)
bm_mix_cell_fracs <- t(bm_mix_cell_fracs)


# ############################################### SAVE RESULTS #######################################################
# ground truth cell fractions for 10 cell types in each patient
# saveRDS(Livnat_cell_fracs, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_cellfracs_no_mix_CV.rds")
saveRDS(bm_mix_cell_fracs, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_cell_fraction_mix.RDS")

# ground truth cell type specific expression in each patient: used for CODEFACS and CIBERSORTx performance evaluation
# saveRDS(Livnat_ct_expr2, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_groundtruth_tpm_no_mix_CV.rds")
saveRDS(bm_mix_ct_expr2, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_cell_type_tpm_mix.RDS")

# pseudo-bulk gene expression for each of the patients
# saveRDS(livnat_bulk, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_bulk_tpm_no_mix_CV.rds")
# saveRDS(bm_mix_bulk, file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_mix.RDS")
write.table(bm_mix_bulk %>% as.data.frame %>% rownames_to_column(var = "GENE"), 
            file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_mix.tsv", 
            sep = "\t", quote = F, row.names = F, col.names = T)
# ############################################### SAVE RESULTS #######################################################


##################################################### mixing experiment 2 (optional) #############################################
## 100 pseudobulk samples where tumor cells are taken from a single patient 
## only but non tumor cells can be taken from multiple patients to create a 
## mixture

set.seed(89128)
## cell types and patients
# cctypes = c("Mal","Endo.","CAF","T.CD8","NK","Macrophage","pDC","skinDC","T.CD4","B.cell")
cctypes  <- meta$celltype %>% unique %>% sort
bm_mix2_celltypes = as.character(meta$celltype)
patients = as.character(meta$patient)
combs <- t(combn(unique(patients[!idx]), 4))
mixes <- sample(seq(nrow(combs)),100)
print(mixes)

## generate cell fractions
bm_mix2_cell_fracs <- sapply(mixes, function(x) {
    counts <- sapply(cctypes, function(y) {
        if(y == "Cancer Epithelial"){
            idx <- which(sapply(combs[x, ], function(p) sum(patients == p & bm_mix2_celltypes == y)) > 0)
            return(sum(patients %in% combs[x, idx[1]] & bm_mix2_celltypes == y))
        }
        return(sum(patients %in% combs[x, ] & bm_mix2_celltypes == y))
    })
    
    counts <-  counts / sum(counts)
    names(counts) <-  cctypes
    return(counts)
})
colnames(bm_mix2_cell_fracs) <- sapply(mixes, function(x) paste("mix", x, sep = ""))

## generate ground truth
pb <- ProgressBar(N = mixes %>% length)
bm_mix2_ct_expr <- lapply(mixes, function(x) {
    pb$tick()
    cdat <- sapply(cctypes, function(y) {
        if(y == "Cancer Epithelial"){
            idx <- which(sapply(combs[x, ], function(p) sum(patients == p & bm_mix2_celltypes == y)) > 0)
            if(sum(patients %in% combs[x, idx[1]] & bm_mix2_celltypes == y) > 1){
                return(rowMeans(sc[, patients %in% combs[x, idx[1]] & bm_mix2_celltypes == y], na.rm = T))
            }
            else if(sum(patients %in% combs[x,idx[1]] & bm_mix2_celltypes == y) == 1){
                return(sc[,patients %in% combs[x,idx[1]] & bm_mix2_celltypes == y])
            }
            else{
                return(rep(0, nrow(sc)))
            }
        }
        
        if(sum(patients %in% combs[x, ] & bm_mix2_celltypes == y) > 1){
            return(rowMeans(sc[, patients %in% combs[x, ] & bm_mix2_celltypes == y], na.rm = T))
        }
        else if(sum(patients %in% combs[x, ] & bm_mix2_celltypes == y) == 1){
            return(sc[, patients %in% combs[x, ] & bm_mix2_celltypes == y])
        }
        else{
            return(rep(0, nrow(sc)))
        }
    })
})
names(bm_mix2_ct_expr) <- sapply(mixes, function(x) paste("mix", x, sep = ""))

## transposing expression profiles
bm_mix2_ct_expr2 <- lapply(cctypes, function(x){
    return(sapply(names(bm_mix2_ct_expr), function(y){
        return(bm_mix2_ct_expr[[y]][, x])
    }))
})
names(bm_mix2_ct_expr2) <- cctypes

## generate expected bulk
bm_mix2_bulk <- sapply(names(bm_mix2_ct_expr), function(x) {
    bb <- bm_mix2_ct_expr[[x]] %*% bm_mix2_cell_fracs[, x]
})
rownames(bm_mix2_bulk) <- rownames(sc)
bm_mix2_cell_fracs <- t(bm_mix2_cell_fracs)


# ############################################### SAVE RESULTS #######################################################
# ground truth cell fractions for 10 cell types in each patient
# saveRDS(Livnat_cell_fracs, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_cellfracs_no_mix_CV.rds")
saveRDS(bm_mix2_cell_fracs, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_cell_fraction_mix2.RDS")

# ground truth cell type specific expression in each patient: used for CODEFACS and CIBERSORTx performance evaluation
# saveRDS(Livnat_ct_expr2, file = "~/forKun/Melanoma_scRNASeq/Livnat_imp_groundtruth_tpm_no_mix_CV.rds")
saveRDS(bm_mix2_ct_expr2, 
        file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_cell_type_tpm_mix2.RDS")

# pseudo-bulk gene expression for each of the patients
# saveRDS(livnat_bulk, file = "~/forKun/Melanoma_scRNASeq/livnat_imp_bulk_tpm_no_mix_CV.rds")
# saveRDS(bm_mix2_bulk, file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_mix2.RDS")
write.table(bm_mix2_bulk %>% as.data.frame %>% rownames_to_column(var = "GENE"), 
            file = "/data/Lab_ruppin/projects/TME_contribution_project/data/SC_data/WuEtAl2021/WuEtAl2021_benchmark_bulk_tpm_mix2.tsv", 
            sep = "\t", quote = F, row.names = F, col.names = T)
# ############################################### SAVE RESULTS #######################################################

