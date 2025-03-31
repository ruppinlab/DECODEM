#### -------------------------------------------------------------------------------
#### created on 03 oct 2022, 07:12pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(rtracklayer)
library(GenomicFeatures) 

pcat <- function(...) cat(paste0(..., "\n"))

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## import gtf file for BrighTNess: https://www.nature.com/articles/s41523-021-00349-y#Sec7
## Illumina HiSeq3000 with 50-bp single-end mode, STAR v2.0.4b, 
## Ensembl release 76 (GRCh38.76)
## download: http://ftp.ensembl.org/pub/release-76/gtf/homo_sapiens/Homo_sapiens.GRCh38.76.gtf.gz
## cmd: curl -o filename URL (change "http" to "ftp")
gtf.file <- "../../data/BrighTNess/Homo_sapiens.GRCh38.76.gtf"
gtf <- import.gff(gtf.file, format = "gtf")

## extract total non-overlapping exon length per gene: https://www.biostars.org/p/83901/
## 1. import the GTF-file that you have also used as input for htseq-count 
## 2. collect the exons per gene id 
## 3. for each gene, reduce all the exons to a set of non overlapping exons, calculate their lengths (widths) and sum 
gtf.txdb <- makeTxDbFromGRanges(gtf, drop.stop.codons = T)
exon.list.gene   <- exonsBy(gtf.txdb, by = "gene", use.names = F)
exon.length.gene <- reduce(exon.list.gene) %>% width %>% sum                        # reduce() & width() are from GenomicRanges/GenomicFeatures


#### -------------------------------------------------------------------------------

## combine gene lengths with annotations.
gene.annot <- as.data.frame(gtf)[, c("gene_id", "gene_name", "gene_biotype")] %>% 
    (function(x) x[!(x$gene_id %>% duplicated), ])
gene.annot <- gene.annot[
    order(gene.annot$gene_id, gene.annot$gene_name, decreasing = T), ]
rownames(gene.annot) <- gene.annot$gene_id

use.genes <- intersect(gene.annot$gene_id, exon.length.gene %>% names)              # genes with annotation
gene.annot[use.genes, "gene_exon_length"] <- exon.length.gene[use.genes]
pcat("\ntotal #genes = ", gene.annot %>% nrow)

## keep only genes with unique ensembl IDs.
gene.annot <- gene.annot[-(gene.annot$gene_name %>% duplicated %>% which), ] %>% 
    `rownames<-`(.$gene_name)
pcat("\ntotal #genes kept = ", gene.annot %>% nrow)


#### -------------------------------------------------------------------------------

## save annotation.
svdat <- F                                                                          # set T to save data 
if (svdat){
    datestamp  <- DateTime()
    annot.file <- sprintf("../../data/BrighTNess/gene_length_ensembl.grch38.76_SRD_%s.txt", 
                          datestamp)
    write.table(gene.annot, file = annot.file, sep = "\t", col.names = T, 
                row.names = F)
}

