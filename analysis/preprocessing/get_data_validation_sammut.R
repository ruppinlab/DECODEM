#### -------------------------------------------------------------------------------
#### created on 02 feb 2023, 06:08pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## load data.
data.path <- "../data/TransNEO_SammutShare/"
data.file <- c("ClinData.csv", 
               "Transneo.diagnostic.DNA.csv", 
               "transneo-diagnosis-DNAseq-mutations.RDS", 
               "transneo-diagnosis-RNAseq-validCounts.RDS", 
               "transneo-diagnosis-RNAseq-validTPM.RDS")

## feature data.
clin.feats.all <- read.table(paste0(data.path, data.file[1]), sep = ",", 
                             header = T, stringsAsFactors = F)
dna.feats.all  <- read.table(paste0(data.path, data.file[2]), sep = ",", 
                             header = T, row.names = 1, stringsAsFactors = F)

## seq data.
dna.data.val <- readRDS(paste0(data.path, data.file[3]))
rna.data.val <- readRDS(paste0(data.path, data.file[4]))
tpm.data.val <- readRDS(paste0(data.path, data.file[5]))

## get training samples & coding genes for filtering.
annot.path <- "../data/TransNEO/use_data/"
annot.file <- c("TransNEO_SupplementaryTablesAll.xlsx", 
                "gene_length_ensembl.grch37.87_SRD_17Mar2022.txt")

clin.train <- openxlsx::read.xlsx(paste0(annot.path, annot.file[1]), sheet = 1, 
                                  startRow = 2, colNames = T, rowNames = T, 
                                  check.names = F)
gene.annot <- read.table(paste0(annot.path, annot.file[2]), sep = "\t", 
                         header = T, row.names = 1, stringsAsFactors = F)

samples.train <- clin.train %>% rownames %>% as.character
gene.annot.pc <- gene.annot %>% filter(gene_biotype == "protein_coding")


#### -------------------------------------------------------------------------------

svdat <- F                                                                          # set T to save data 

## process validation RNA data.
fcat("\nvalidation data size: {tpm.data.val %>% nrow} x {tpm.data.val %>% ncol}")

## keep protein-coding genes only.
keep.genes <- intersect(gene.annot.pc$gene_name, tpm.data.val %>% rownames)
tpm.data.val.pc <- tpm.data.val[keep.genes, ] %>% as.matrix                         # allows duplicate rownames if exist
rownames(tpm.data.val.pc) <- tpm.data.val.pc %>% rownames %>% toupper
genes.dup <- rownames(tpm.data.val.pc) %>% table %>% 
    (function(freq) freq[freq > 1] %>% names) %>%                                   # duplicated gene symbols
    lapply(function(gn) which(rownames(tpm.data.val.pc) == gn)) %>% 
    Reduce(f = union)                                                               # duplicate symbol indices
tpm.data.val.pc <- tpm.data.val.pc[-genes.dup, ] %>% as.data.frame %>%              # remove duplicates
    rownames_to_column(var = "GENE")

fcat("kept protein-coding genes only: m = {tpm.data.val.pc %>% nrow}")

if (svdat){
    fcat("\nsaving validation TPM data ({tpm.data.val.pc %>% nrow} x {tpm.data.val.pc %>% ncol})...")
    
    out.file <- "transneo-validation-TPM-coding-genes_v2.txt"
    write.table(tpm.data.val.pc, file = paste0(data.path, out.file), sep = "\t", 
                row.names = F, col.names = T)
    
    fcat("done!")
}


#### -------------------------------------------------------------------------------

svdat <- F                                                                          # set T to save data 

## process clinical features listed below.
## Age.at.diagnosis, Histology, ER.status, HER2.status, LN.at.diagnosis, 
## Grade.pre.chemotherapy, Size.at.diagnosis


## remove repeated sample "T186" that has been repeated with different treatment 
## info- presumably due to some error- since Supplementary Table 5 only reports one
## single treatment regimen: "T-EC + Trastuzumab + Pertuzumab"
clin.feats.filt <- ((clin.feats.all$Trial.ID == "T186") & 
    (clin.feats.all$Regimen.Name != "T-EC + Trastuzumab + Pertuzumab")) %>% 
    (function(cond) clin.feats.all[-which(cond), ] %>% `rownames<-`(NULL)) %>% 
    column_to_rownames("Trial.ID") 


## age, tumor grade: already integers- nothing to change.

## tumor size: replace "unevaluable" with 132mm. 
## "Patients who had a clinically unevaluable tumour size was assumed to have a 
## volume 10% larger than the largest present in the cohort (see methods) – I would 
## suggest that you set this to 132 mm" (sammut et al.)
## 'imaging.size' is the pretherapy tumor size.
clin.feats.filt$Size.at.diagnosis <- clin.feats.filt$Imaging.size %>% 
    replace(list = (clin.feats.filt$Imaging.size == "Unevaluable") %>% which, 
            values = 132) %>% as.integer                                            # max.tumor.size = 120


## histology: 1 if contains IDC, 0 otherwise (see methods).
clin.feats.filt$Histology.original <- clin.feats.filt$Histology                     # preserve the original data
clin.feats.filt$Histology <- clin.feats.filt$Histology %>% 
    grepl(pattern = "IDC") %>% as.integer


## ER, HER2, LN status: replace POS with +1, NEG with -1.
feats <- c("ER.status", "HER2.status", "LN.at.diagnosis")
clin.feats.filt[paste(feats, "original", sep = ".")] <- clin.feats.filt[feats]      # preserve the original data
clin.feats.filt[feats] <- clin.feats.filt[feats] %>% apply(MARGIN = 2, function(x){
    x %>% gsub(pattern = "POS", replacement = +1) %>% 
        gsub(pattern = "NEG", replacement = -1) %>% as.integer
})


## divide data into training & validation cohorts. 
samples.trn <- intersect(clin.feats.filt %>% rownames, samples.train)
samples.val <- setdiff(clin.feats.filt %>% rownames, samples.trn)
clin.feats.trn <- clin.feats.filt[samples.trn, ]
clin.feats.val <- clin.feats.filt[samples.val, ]

fcat("\nclinical features for modeling:")
fcat("training size = {clin.feats.trn %>% nrow} x {clin.feats.trn %>% ncol}")
fcat("validation size = {clin.feats.val %>% nrow} x {clin.feats.val %>% ncol}")

if (svdat){
    fcat("\nsaving clinical features in xlsx...")
    
    out.file <- "transneo-diagnosis-clinical-features.xlsx"
    out.data <- list("training" = clin.feats.trn %>% rownames_to_column("Trial.ID"), 
                     "validation" = clin.feats.val %>% rownames_to_column("Trial.ID"))
    WriteXlsx(out.data, file.name = paste0(data.path, out.file), verbose = T, 
              row.names = F, col.names = T)
    
    fcat("done!")
}


#### -------------------------------------------------------------------------------

svdat <- F                                                                          # set T to save data 

## prepare DNA features.
dna.feats.all$HLA.LOH <- dna.feats.all$LOH.HLA                                      # to be consistent with 'training_df.csv'
dna.feats.all$Expressed.NAg <- dna.feats.all$Expressed.NAg %>% is.na %>% 
    replace(x = dna.feats.all$Expressed.NAg, values = 0)                            # replace missing NAg values (n = 7) with 0

## separate for training & validation cohorts.
samples.trn <- intersect(dna.feats.all %>% rownames, samples.train)
samples.val <- setdiff(dna.feats.all %>% rownames, samples.trn)
dna.feats.trn <- dna.feats.all[samples.trn, ]
dna.feats.val <- dna.feats.all[samples.val, ]

if (svdat){
    fcat("\nsaving DNA features...")
    fcat("training: {dna.feats.trn %>% nrow} x {dna.feats.trn %>% ncol}")
    fcat("validation: {dna.feats.val %>% nrow} x {dna.feats.val %>% ncol}")
    
    out.file <- c("transneo-diagnosis-DNA-features.txt", 
                  "transneo-validation-DNA-features.txt")
    out.data <- list(dna.feats.trn %>% rownames_to_column("Trial.ID"), 
                     dna.feats.val %>% rownames_to_column("Trial.ID"))
    
    plyr::l_ply(1:2, function(nn){                                                  # returns nothing
        write.table(out.data[[nn]], file = paste0(data.path, out.file[nn]), 
                    sep = "\t", col.names = T, row.names = F)
    })
    
    fcat("done!")
}

