#### -------------------------------------------------------------------------------
#### created on 03 oct 2022, 07:22pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

log.to.rpkm <- function(log.counts, base = 2, offset = 1){
    read.rpkm <- base ^ log.counts - offset                                         # inverse log and subtract the offset
}

rpkm.to.tpm <- function(read.rpkm){                                                 # per sample TPM conversion
    read.tpm <- read.rpkm / sum(read.rpkm) * 1e6
}

pcat <- function(...) cat(paste0(..., "\n"))

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## from Mertzer-Filho et al.: https://www.nature.com/articles/s41523-021-00349-y#Sec7 
## "Samples with >10 million unique reads were included for further analyses as 
## Reads per Kilobase per Million Reads (RPKM). RNA-seq reads were aligned to the 
## Ensembl release 76 top-level assembly with STAR version 2.0.4b. Gene counts were 
## derived from the number of uniquely aligned unambiguous reads by 
## Subread:featureCount version 1.4.5. Transcript counts were produced by Sailfish 
## version 0.6.3. All gene-level and transcript counts were then imported into the 
## R/Bioconductor package EdgeR and TMM normalization size factors were calculated 
## to adjust samples for differences in library size, resulting in RPKM which were 
## used in downstream analyses. Genes or transcripts not expressed in any sample or 
## less than one count-per-million in the minimum group size minus one were 
## excluded from further analysis."

## read RNA-seq RPKM data.
pcat("\nreading RPKM data...")
count.file    <- "../data/BrighTNess/GSE164458_BrighTNess_RNAseq_log2_Processed_ASTOR.txt"
rnaseq.rpkm <- read.table(count.file, sep = "\t", header = T, row.names = 1, 
                          check.names = F) %>% 
    `colnames<-`(colnames(.) %>% sapply(function(smpl) paste0("Sample_", smpl)))    # make columns as 'Sample_{sample_id}'
cat("removing log2 normalization...\n")
rnaseq.rpkm <- rnaseq.rpkm %>% log.to.rpkm(offset = 1)                              # remove log normalization
pcat("total gene count = ", rnaseq.rpkm %>% nrow)

## filter low count genes. 
## for multiple gene symbol mapping issue: https://www.biostars.org/p/389804/
low.count.th <- 100
pcat("\nfiltering genes with low read count (Th = ", low.count.th, ")...")
rnaseq.rpkm.filt <- rnaseq.rpkm[
    which(rnaseq.rpkm %>% rowSums > low.count.th), ]
pcat("gene count after filtering = ", rnaseq.rpkm.filt %>% nrow)


## read gene annotations [use gene symbols as rows as counts are at gene level].
pcat("\nreading gene annotations...")
annot.file <- "../data/BrighTNess/gene_length_ensembl.grch38.76_SRD_03Oct2022.txt"
gene.annot <- read.table(annot.file, header = T) %>% `rownames<-`(.$gene_name)

## keep only the genes with annotations.
use.genes        <- intersect(rnaseq.rpkm.filt %>% rownames, gene.annot$gene_name)
rnaseq.rpkm.filt <- rnaseq.rpkm.filt[use.genes, ]
gene.annot.filt  <- gene.annot[use.genes, ]
pcat("\ngene count with annotation = ", rnaseq.rpkm.filt %>% nrow)


#### -------------------------------------------------------------------------------

## convert counts to TPM.
pcat("\nconverting counts to TPM...")
pb <- ProgressBar(N = rnaseq.rpkm.filt %>% ncol)                                    # progress bar
rnaseq.tpm <- rnaseq.rpkm.filt %>% apply(MARGIN = 2, function(sample.rpkm){ 
    pb$tick()
    rpkm.to.tpm(read.rpkm = sample.rpkm)
})
pcat("done!")

## sanity check.
rnaseq.tpm <- as.data.frame(rnaseq.tpm)
pcat("checking if there is any missing value... ", rnaseq.tpm %>% is.na %>% any)
pcat("checking if sample-wise values sum up to 1M... ", 
     all(abs(rnaseq.tpm %>% colSums - 1e6) < 1e-4))


#### -------------------------------------------------------------------------------

## use only protein-coding genes.
pcat("\nusing only protein-coding genes...")
genes.pc <- intersect(
    gene.annot$gene_name[gene.annot$gene_biotype == "protein_coding"], 
    rnaseq.tpm %>% rownames
)
rnaseq.tpm <- rnaseq.tpm[genes.pc, ]

## check gene names.
gene.symbols <- rnaseq.tpm %>% rownames %>% toupper
if (gene.symbols %>% unique %>% length == rnaseq.tpm %>% nrow){
    pcat("\nconvert gene symbols to uppercase")
    rownames(rnaseq.tpm) <- gene.symbols
} else{
    stop("all genes are NOT unique!")
}

pcat("done! final gene count = ", rnaseq.tpm %>% nrow)


#### -------------------------------------------------------------------------------

## save data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    cat("\nsaving file... ")
    datestamp <- DateTime()
    tpm.file  <- sprintf("../../data/BrighTNess/GSE164458_BrighTNess_RNAseq_TPM_v2_SRD_%s.csv", 
                         datestamp)
    write.table(x = cbind(GENE = rownames(rnaseq.tpm) %>% toupper, rnaseq.tpm), 
                file = tpm.file, sep = ",", col.names = T, row.names = F)           # save gene as a column
    pcat("done!")
}


