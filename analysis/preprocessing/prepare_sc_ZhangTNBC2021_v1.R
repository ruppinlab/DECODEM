#### -------------------------------------------------------------------------------
#### created on 12 apr 2023, 04:40pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(Matrix)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## get data & annotations.
data.paths <- c("../data/SC_data/ZhangTNBC2021/", 
                "../data/TransNEO/use_data/")
data.files <- c("TNBC_scExp_resp_pre_ZhangEtAl2021.RDS", 
                "TNBC_scData_ZhangEtAl2021.xlsx", 
                "gene_length_ensembl.grch37.87_SRD_17Mar2022.txt")

tnbc.data  <- readRDS(paste0(data.paths[1], data.files[1]))

## use TransNEO annotations (hg19) to filter protein-coding genes.
gene.annot <- read.table(paste0(data.paths[2], data.files[3]), sep = "\t", 
                         header = T, as.is = T)
gene.annot.pc <- gene.annot %>% filter(gene_biotype == "protein_coding") %>% 
    (function(df) df[!duplicated(df$gene_name), ]) %>% `rownames<-`(.$gene_name)

## patient/sample lists by treatment + sample type.
sample.type <- "tissue"                                                             # use "tissue" samples (other option: "blood")
sample.type <- sample.type %>% substr(1, 1) %>% `names<-`(sample.type)
sc.samples  <- tnbc.data$patients$Treatment %>% unique %>% 
    sapply(simplify = F, function(trt){
        tnbc.data$patients %>% filter(Treatment == trt, Origin == sample.type) %>% 
            .$Sample.id
    })


## rename cell types to match with TransNEO.
## B cell: B-cells, T cell: T-cells, Myeloid cell: Myeloid, ILC cell: ILC
cells.new <- tnbc.data$cells %>% sapply(function(ctp){
    strsplit(ctp, split = " ")[[1]] %>% (function(x){ 
        ifelse((x[1] %>% str_length) == 1, 
               yes = paste(x[1], "cells", sep = "-"),                               # B cell / T cell
               no = x[1])
    })
})


#### -------------------------------------------------------------------------------

## relevant functions.
## rename cell types to match with deconvolved data.
rename.cell.type <- function(ctp.list){
    ctp.list.new <- ctp.list
    for (ctp in cells.new %>% names){
        ctp.list.new <- ctp.list.new %>% 
            gsub(pattern = ctp, replacement = cells.new[ctp])
    }
    ctp.list.new
}

## SC data is from 10X - in log(1 + x) format, where sum(x) = 1e4.
## reverse the log-transform & multiply by 100 for TPM equivalence.
tpm.from.10x <- function(expr.10x){
    expr.10x.tpm <- ((exp(expr.10x) - 1) * 100) %>% as.matrix %>% as.data.frame
    expr.10x.tpm
}


#### -------------------------------------------------------------------------------

## get chemotherapy data.
treat <- "Chemo"

samples.cm <- sc.samples[[treat]]
annot.cm   <- tnbc.data$annot %>% 
    filter(Treatment == treat, Origin == sample.type) %>% 
    mutate(Cell.type = .$Cluster_org %>% rename.cell.type)

pb <- ProgressBar(N = tnbc.data$cells %>% length)
sc.exp.cm <- tnbc.data$cells %>% sapply(simplify = F, function(ctp){
    pb$tick()
    
    ## get relevant cell ids.
    annot.ctp  <- annot.cm %>% filter(Cluster_org == ctp)
    cells.ctp  <- samples.cm %>% lapply(function(smpl){
        annot.ctp %>% filter(Sample.id == smpl) %>% .$Cell.id                       # cell ids per patient sample
    }) %>% Reduce(f = union)
    
    ## keep protein-coding genes only.
    sc.exp.ctp <- tnbc.data$tissue.norm[[ctp]]                                      # use normalized sc exp
    genes.ctp  <- sc.exp.ctp %>% rownames %>% intersect(., gene.annot.pc$gene_name)
    sc.exp.ctp <- sc.exp.ctp[genes.ctp, cells.ctp] %>% tpm.from.10x                 # convert to tpm
    sc.exp.ctp
})
names(sc.exp.cm) <- cells.new


#### -------------------------------------------------------------------------------

## get immunotherapy data.
treat <- "Anti-PD-L1+Chemo"

samples.im <- sc.samples[[treat]]
annot.im   <- tnbc.data$annot %>% 
    filter(Treatment == treat, Origin == sample.type) %>% 
    mutate(Cell.type = .$Cluster_org %>% rename.cell.type)

pb <- ProgressBar(N = tnbc.data$cells %>% length)
sc.exp.im <- tnbc.data$cells %>% sapply(simplify = F, function(ctp){
    pb$tick()
    
    ## get relevant cell ids.
    annot.ctp  <- annot.im %>% filter(Cluster_org == ctp)
    cells.ctp  <- samples.im %>% lapply(function(smpl){
        annot.ctp %>% filter(Sample.id == smpl) %>% .$Cell.id                       # cell ids per patient sample
    }) %>% Reduce(f = union)
    
    ## keep protein-coding genes only.
    sc.exp.ctp <- tnbc.data$tissue.norm[[ctp]]                                      # use normalized sc exp
    genes.ctp  <- sc.exp.ctp %>% rownames %>% intersect(., gene.annot.pc$gene_name)
    sc.exp.ctp <- sc.exp.ctp[genes.ctp, cells.ctp] %>% tpm.from.10x
    sc.exp.ctp
})
names(sc.exp.im) <- cells.new


#### -------------------------------------------------------------------------------

## save data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    ## save chemo data.
    out.file <- glue("TNBC_scExp_{sample.type %>% names}_pre_Chemo_ZhangEtAl2021.tsv")
    pb <- ProgressBar(N = cells.new %>% length)
    cells.new %>% plyr::l_ply(function(ctp){
        pb$tick()
        out.file.ctp <- out.file %>% 
            gsub(pattern = ".tsv", replacement = glue("_{ctp}.tsv"))
        write.table(sc.exp.cm[[ctp]], file = paste0(data.paths[1], out.file.ctp), 
                    sep = "\t", col.names = T, row.names = T)
    })
    # WriteXlsx(sc.exp.cm, file.name = paste0(data.paths[1], out.file))
    
    ## save anti-PD-L1+chemo data.
    out.file <- glue("TNBC_scExp_{sample.type %>% names}_pre_Anti-PD-L1_Chemo_ZhangEtAl2021.tsv")
    pb <- ProgressBar(N = cells.new %>% length)
    cells.new %>% plyr::l_ply(function(ctp){
        pb$tick()
        out.file.ctp <- out.file %>% 
            gsub(pattern = ".tsv", replacement = glue("_{ctp}.tsv"))
        write.table(sc.exp.im[[ctp]], file = paste0(data.paths[1], out.file.ctp), 
                    sep = "\t", col.names = T, row.names = T)
    })
    # WriteXlsx(sc.exp.im, file.name = paste0(data.paths[1], out.file))
    
    ## save annotations.
    out.file <- glue("TNBC_scAnnot_{sample.type %>% names}_pre_ZhangEtAl2021.xlsx")
    out.obj  <- list("Chemo" = annot.cm, "Anti-PD-L1_Chemo" = annot.im)
    WriteXlsx(out.obj, file.name = paste0(data.paths[1], out.file), 
              col.names = T, row.names = F)
}


