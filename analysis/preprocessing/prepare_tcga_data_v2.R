#### ---------------------------------------------------------------------------
#### @author: dhrubas2
#### created on: 2024-09-06 13:52:19 EDT
#### ---------------------------------------------------------------------------

.wpath. <- "/data/Lab_ruppin/projects/TME_contribution_project/analysis/analysis_final/"
.mpath. <- "/home/dhrubas2/vivid/miscellaneous.R"

setwd(.wpath.)
source(.mpath.)

library(reticulate)

read.data <- function(path, sep = "\t", header = T, row.names = NA){            # fast data reading function
    data <- data.table::fread(path, sep = sep, header = header, data.table = F)
    if (!(row.names %>% is.na)){
        data <- data %>% column_to_rownames(data %>% colnames %>% .[row.names])
    }
    data
}

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                        # f-string print akin to python

cat("\014")                                                                     # clears console


#### ---------------------------------------------------------------------------

data.path  <- list("bulk"  = "../../data/TCGA/", 
                   "decon" = "../../data/TCGA/out_codefacs_tcga/")

data.files <- list("clin"  = "TCGA_BRCA_clinical_data_combined.tsv", 
                   "surv"  = "brca_tcga_pan_can_atlas_2018_survival_curated.txt", 
                   "bulk"  = "TCGA_BRCA_htseq_TPM_matched2.tsv", 
                   "decon" = c(), 
                   "frac"  = "estimated_cell_fractions.txt", 
                   "conf"  = "confidence_score.txt")
data.files$decon = data.path$decon %>% list.files(pattern = "expression")


#### ---------------------------------------------------------------------------

## get bulk expression.
exp.bulk  <- read.data(paste0(data.path$bulk, data.files$bulk), row.names = 1)


## get clinical data.
clin.info <- read.data(paste0(data.path$bulk, data.files$clin))
keep.cols <- c("Patient ID", "Sample ID", "Sample Type", "Diagnosis Age", "Sex", 
               "Neoplasm Disease Stage American Joint Committee on Cancer Code", 
               "HER2 Status By IHC", "ER Status By IHC", "PR status by ihc", 
               "PAM50 Subtype", "TMB (nonsynonymous)", "MSIsensor Score", 
               "MSI MANTIS Score", "Mutation Count", "Fraction Genome Altered")

clin.data <- clin.info[, keep.cols] %>% `colnames<-`(
    c("Patient_ID", "Sample_ID", "Sample_type", "Age", "Sex", "Stage", 
      "HER2_status", "ER_status", "PR_status", "PAM50_subtype", "TMB", 
      "MSI_score_sensor", "MSI_score_MANTIS", "Mutation_count", 
      "Fraction_genome_altered")) %>% `rownames<-`(.$Sample_ID)


## get survival data.
surv.info <- read.data(paste0(data.path$bulk, data.files$surv))

surv.data <- surv.info %>% select(`_PATIENT`, sample, OS, OS.time, PFI, 
                                  PFI.time, DFI, DFI.time, DSS, DSS.time) %>% 
    `colnames<-`(c("Patient_ID", "Sample_ID", "OS", "OS_time", "PFI", 
                   "PFI_time", "DFI", "DFI_time", "DSS", "DSS_time")) %>% 
    `rownames<-`(.$Sample_ID)


#### ---------------------------------------------------------------------------

## get deconvolved data.
cell.frac  <- read.data(paste0(data.path$decon, data.files$frac), row.names = 1)

conf.score <- read.data(paste0(data.path$decon, data.files$conf), row.names = 1)

cell.types <- data.files$decon %>% sapply(USE.NAMES = F, function(fn){          # get cell types
    fn %>% str_split(pattern = "_", simplify = T) %>% .[2] %>% 
        strsplit(split = ".", fixed = T) %>% .[[1]] %>% .[1]}) %>% 
    gsub(pattern = " ", replacement = "_")

pb <- ProgressBar(N = data.files$decon %>% length)
exp.decon  <- data.files$decon %>% sapply(simplify = F, function(decon.file){
    pb$tick()
    read.data(paste0(data.path$decon, decon.file), sep = " ", row.names = 1)
}) %>% `names<-`(cell.types)


## get full datasets.
exp.data  <- exp.decon;     exp.data$Bulk <- exp.bulk


#### ---------------------------------------------------------------------------

cell.types <- c(cell.types, "Bulk")

## subset data by HER2 status.
samples.all <- Reduce(
    list(exp.bulk %>% colnames, clin.data$Sample_ID, surv.data$Sample_ID), 
    f = intersect)

samples.hp  <- intersect(
    clin.data %>% filter(HER2_status == "Positive") %>% rownames, 
    samples.all)

samples.hn  <- intersect(
    clin.data %>% filter(HER2_status == "Negative") %>% rownames, 
    samples.all)


data.all <- list(
    "exp"  = cell.types %>% sapply(simplify = F, function(ctp){
        exp.data[[ctp]][, samples.all] }), 
    "resp" = surv.data[samples.all, ], "frac" = cell.frac[samples.all, ], 
    "conf" = conf.score,               "clin" = clin.data[samples.all, ]
)

data.hp  <- list(
    "exp"  = cell.types %>% sapply(simplify = F, function(ctp){
        exp.data[[ctp]][, samples.hp] }), 
    "resp" = surv.data[samples.hp, ], "frac" = cell.frac[samples.hp, ], 
    "conf" = conf.score,               "clin" = clin.data[samples.hp, ]
)

data.hn  <- list(
    "exp"  = cell.types %>% sapply(simplify = F, function(ctp){
        exp.data[[ctp]][, samples.hn] }), 
    "resp" = surv.data[samples.hn, ], "frac" = cell.frac[samples.hn, ], 
    "conf" = conf.score,               "clin" = clin.data[samples.hn, ]
)


#### ---------------------------------------------------------------------------

svdat <- T

if (svdat){
    out.path <- "../../data/TransNEO/transneo_analysis/"
    
    ## all.
    out.file <- "tcga_brca_data_surv_all.pkl"
    py_save_object(data.all, filename = paste0(out.path, out.file), 
                   pickle = "pickle")
    fcat(out.file)
    
    
    ## HER2-positive.
    out.file <- "tcga_brca_data_surv_her2pos.pkl"
    py_save_object(data.hp, filename = paste0(out.path, out.file), 
                   pickle = "pickle")
    fcat(out.file)
    
    out.file <- "tcga_brca_data_surv_her2neg.pkl"
    py_save_object(data.hn, filename = paste0(out.path, out.file), 
                   pickle = "pickle")
    fcat(out.file)
}









