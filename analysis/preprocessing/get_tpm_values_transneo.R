#### -------------------------------------------------------------------------------
#### created on 17 mar 2022, 05:52pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

pcat <- function(...) cat(paste0(..., "\n"))

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## import rna-seq count data & gene annotation.
cat("\nreading raw counts...\n")
count.file <- "../data/TransNEO/use_data/transneo-diagnosis-RNAseq-rawcounts.tsv"
rnaseq.counts <- read.table(count.file, header = T, row.names = 1)
pcat("total gene count = ", rnaseq.counts %>% nrow)

cat("\nreading gene lengths...\n")
annot.file <- "../data/TransNEO/use_data/gene_length_ensembl.grch37.87_SRD_17Mar2022.txt"
gene.annot <- read.table(annot.file, header = T) %>% `rownames<-`(.$gene_id)


#### -------------------------------------------------------------------------------

## filter low count genes. 
## for multiple gene symbol mapping issue: https://www.biostars.org/p/389804/
low.count.th <- 100
pcat("\nfiltering genes with low read count (Th = ", low.count.th, ")...")
rnaseq.counts.filt <- rnaseq.counts[
    which(rnaseq.counts %>% rowSums > low.count.th), ]
pcat("gene count after filtering = ", rnaseq.counts.filt %>% nrow)

## keep only genes with annotations.
use.genes <- intersect(rnaseq.counts.filt %>% rownames, gene.annot$gene_id)
rnaseq.counts.filt <- rnaseq.counts.filt[use.genes, ]
pcat("\ngene count with annotation = ", rnaseq.counts.filt %>% nrow)


#### -------------------------------------------------------------------------------

## convert counts to TPM.
count.to.tpm <- function(read.counts, gene.lengths){                                # per sample TPM conversion
    rate <- read.counts / gene.lengths
    read.tpm <- rate / sum(rate) * 1e6
}

cat("\nconverting counts to tpm...\n")
pb <- ProgressBar(N = rnaseq.counts %>% ncol)                                       # progress bar
rnaseq.tpm <- rnaseq.counts.filt[use.genes, ] %>% 
    apply(MARGIN = 2, function(sample.counts){ 
        pb$tick()
        count.to.tpm(read.counts = sample.counts, 
                     gene.lengths = gene.annot[use.genes, "gene_exon_length"])
    })
cat("done!\n")

# sanity check.
pcat("checking if sample-wise values sum up to 1M...", 
     all(abs(rnaseq.tpm %>% colSums - 1e6) < 1e-4))


#### -------------------------------------------------------------------------------

## convert ensembl IDs to gene symbols & remove duplicates.
cat("\nconverting ensembl ids to gene symbols...\n")
rownames(rnaseq.tpm) <- gene.annot[rownames(rnaseq.tpm), "gene_name"] %>% toupper
genes.dup <- rownames(rnaseq.tpm) %>% table %>% 
    (function(freq) freq[freq > 1] %>% names) %>%                                   # duplicated gene symbols
    lapply(function(gn) which(rownames(rnaseq.tpm) == gn)) %>% 
    Reduce(f = union)                                                               # duplicate symbol indices
rnaseq.tpm <- rnaseq.tpm[-genes.dup, ]                                              # remove duplicates

pcat("\nchecking if duplicate gene symbols are present... ", 
     rnaseq.tpm %>% rownames %>% anyDuplicated %>% as.logical)

rnaseq.tpm <- as.data.frame(rnaseq.tpm)

## use only protein-coding genes.
cat("\nusing only protein-coding genes...\n")
genes.pc <- intersect(
    gene.annot$gene_name[gene.annot$gene_biotype == "protein_coding"], 
    rnaseq.tpm %>% rownames
)
rnaseq.tpm <- rnaseq.tpm[genes.pc, ]
pcat("done! final gene count = ", rnaseq.tpm %>% nrow)


#### -------------------------------------------------------------------------------

## save data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    cat("\nsaving file... ")
    tpm.file <- sprintf("../data/TransNEO/use_data/transneo-diagnosis-RNAseq-TPM_SRD_%s.tsv", DateTime())
    write.table(x = cbind(Gene = rownames(rnaseq.tpm), rnaseq.tpm), 
                file = tpm.file, sep = "\t", col.names = T, row.names = F)          # save gene as a column
    cat("done!\n")
}

