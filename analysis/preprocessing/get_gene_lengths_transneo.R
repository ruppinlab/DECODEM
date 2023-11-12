#### -------------------------------------------------------------------------------
#### created on 17 mar 2022, 03:21pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(rtracklayer)
library(GenomicFeatures) 

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## import gtf file for Sammut et al.: https://www.nature.com/articles/s41586-021-04278-5#Sec10
## Illumina HiSeq4000 system run in 75-bp paired-end mode, STAR v2.5.2b67, 
## GRCh37, Ensembl release 87 (GRCh37.87)
## download: http://ftp.ensembl.org/pub/grch37/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz
## cmd: curl -o filename URL (change "http" to "ftp")
gtf.file <- "../data/TransNEO/use_data/Homo_sapiens.GRCh37.87.gtf"
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


#### -------------------------------------------------------------------------------

## save annotation.
svdat <- F                                                                          # set T to save data 
if (svdat){
    annot.file <- sprintf("../data/TransNEO/use_data/gene_length_ensembl.grch37.87_SRD_%s.txt", DateTime())
    write.table(gene.annot, file = annot.file, sep = "\t", col.names = T, 
                row.names = F)
}

