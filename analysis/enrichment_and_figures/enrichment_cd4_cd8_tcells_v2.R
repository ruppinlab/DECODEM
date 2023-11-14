#### -------------------------------------------------------------------------------
#### created on 24 aug 2023, 02:41pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(GSVA)
library(PRROC)
library(rstatix)
library(ggpubr)


fcat <- function(...) cat(paste0(glue(...), "\n"))

keep.first.str <- function(strs.with.delim, delim = "."){
    strs.with.delim %>% sapply(function(str.with.delim){
        strsplit(str.with.delim, split = delim, fixed = T)[[1]][1]
    })
}

read.data <- function(file, sep = "\t", sheet = 1, header = T, index = T, skip = 0){
    if (file %>% grepl(pattern = ".xlsx")){
        openxlsx::read.xlsx(file, sheet = sheet, startRow = skip + 1, 
                            colNames = header, rowNames = (index > 0), 
                            check.names = F)
    } else{
        read.table(file, sep = sep, skip = skip, as.is = T, header = header, 
                   row.names = if(index) 1 else NULL, check.names = F)
    }
}

mmscale <- function(x) (x - x %>% min) / (x %>% range %>% diff)

classifier.performance <- function(data, label, pred, method = "integral"){         # compute average precision (method = integral / davis.goodrich)
    if (data %>% missing){ true <- label;   pred <- pred }
    else { true <- data[[label]];   pred <- data[[pred]] }
    if (true %>% is.factor) true <- (true %>% levels %>% as.numeric)[true]
    if (pred %>% is.factor) pred <- (pred %>% levels %>% as.numeric)[pred]
    
    perf <- c("AUC" = roc.curve(scores.class0 = pred, weights.class0 = true)$auc, 
              "AP" = pr.curve(scores.class0 = pred, weights.class0 = true)[[
                  glue("auc.{method}")]])
    perf
}

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## read sc DEGs & exp/resp data.
data.path <- c("../data/celltype_signature/", 
               "../data/TransNEO/CODEFACS_results/", 
               "../data/TransNEO/use_data/", 
               "../data/TransNEO_SammutShare/out_codefacs_tn_val_v2/", 
               "../data/TransNEO_SammutShare/", 
               "../data/BrighTNess/out_codefacs_brightness_v2/", 
               "../data/BrighTNess/")

data.file <- c("BRCA_GSE176078_AllDiffGenes_table.tsv", 
               "expression_T-cells.txt", 
               "transneo-diagnosis-RNAseq-TPM_SRD_26May2022.tsv", 
               "TransNEO_SupplementaryTablesAll.xlsx", 
               "expression_T-cells.txt", 
               "transneo-validation-TPM-coding-genes_v2.txt", 
               "transneo-diagnosis-clinical-features.xlsx", 
               "expression_T-cells.txt", 
               "GSE164458_BrighTNess_RNAseq_TPM_v2_SRD_09Oct2022.csv", 
               "GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx")


## get CD4 / CD8 T-cells DEG signatures.
sc.DEGs  <- read.data(paste0(data.path[1], data.file[1]), index = F)

cut.FC   <- 0.5                                                                     # cut-off for log2FC => FC = post/pre = 2^0.5 = 1.41
lineages <- c("CD4" = "CD4Tconv", "CD8" = "CD8Tex")

sign.tcells <- lineages %>% names %>% sapply(simplify = F, function(ctp){
    sc.DEGs %>% filter(Celltype_major_lineage == lineages[ctp], 
                       log2FC >= cut.FC) %>% 
        dplyr::arrange(desc(log2FC)) %>% .$Gene %>% unique
})

fcat("#signature genes:");  print(sign.tcells %>% sapply(length))
fcat("#common genes = {Reduce(intersect, sign.tcells) %>% length}")


## TransNEO - keep data for chemotherapy alone.
exp.tcells.tn <- read.data(paste0(data.path[2], data.file[2]), sep = " ")
exp.bulk.tn   <- read.data(paste0(data.path[3], data.file[3]))
clin.info.tn  <- read.data(paste0(data.path[3], data.file[4]), skip = 1)

clin.cm.tn  <- clin.info.tn %>% filter(
    rownames(clin.info.tn) %in% colnames(exp.tcells.tn), 
    RCB.category %>% is.na %>% `!`, 
    aHER2.cycles %>% is.na)

resp.cm.tn <- clin.cm.tn %>% rownames %>% sapply(function(smpl) 
    ifelse(clin.info.tn[smpl, "pCR.RD"] == "pCR", 1, 0))

exp.tcells.cm.tn <- exp.tcells.tn[, clin.cm.tn %>% rownames]
exp.bulk.cm.tn   <- exp.bulk.tn[, clin.cm.tn %>% rownames]


## ARTemis + PBCP - keep data for chemotherapy alone.
exp.tcells.tn.val <- read.data(paste0(data.path[4], data.file[5]), sep = " ")
exp.bulk.tn.val   <- read.data(paste0(data.path[5], data.file[6]))
clin.info.tn.val  <- read.data(paste0(data.path[5], data.file[7]), sheet = 2)

clin.cm.tn.val  <- clin.info.tn.val %>% filter(
    rownames(clin.info.tn.val) %in% colnames(exp.tcells.tn.val), 
    pCR.RD %>% is.na %>% `!`, 
    anti.her2.cycles %>% is.na)

resp.cm.tn.val <- clin.cm.tn.val %>% rownames %>% sapply(function(smpl) 
    ifelse(clin.info.tn.val[smpl, "pCR.RD"] == "pCR", 1, 0))

exp.tcells.cm.tn.val <- exp.tcells.tn.val[, clin.cm.tn.val %>% rownames]
exp.bulk.cm.tn.val   <- exp.bulk.tn.val[, clin.cm.tn.val %>% rownames]


## BrighTNess - keep data for arm B alone.
exp.tcells.bn <- read.data(paste0(data.path[6], data.file[8]), sep = " ")
exp.bulk.bn   <- read.data(paste0(data.path[7], data.file[9]), sep = ",")
clin.info.bn  <- read.data(paste0(data.path[7], data.file[10]))

clin.cm.bn  <- clin.info.bn %>% filter(
    rownames(clin.info.bn) %in% colnames(exp.tcells.bn), 
    residual_cancer_burden_class %>% is.na %>% `!`, 
    planned_arm_code == "B")

resp.cm.bn <- clin.cm.bn %>% rownames %>% sapply(function(smpl) 
    ifelse(clin.info.bn[smpl, "pathologic_complete_response"] == "pCR", 1, 0))

exp.tcells.cm.bn <- exp.tcells.bn[, clin.cm.bn %>% rownames]
exp.bulk.cm.bn   <- exp.bulk.bn[, clin.cm.bn %>% rownames]

datasets <- c("TransNEO", "ARTemis + PBCP", "BrighTNess")
fcat("\ndataset sizes = ");     print(
    list(exp.tcells.cm.tn, exp.tcells.cm.tn.val, exp.tcells.cm.bn) %>% 
        sapply(dim) %>% `dimnames<-`(list(c("genes", "samples"), datasets)) %>% t)


#### -------------------------------------------------------------------------------

## enrichment analysis for T-cells expression.
use.ctp  <- "T-cells"                                                               # T-cells / Bulk

data.all <- datasets %>% sapply(simplify = F, function(ds){
    ## get relevant data.
    if (ds %>% tolower == "transneo"){
        exp.data  <- if (use.ctp == "T-cells") exp.tcells.cm.tn else exp.bulk.cm.tn
        resp.data <- resp.cm.tn
    } else if (ds %>% tolower == "artemis + pbcp"){
        exp.data  <- if (use.ctp == "T-cells") 
            exp.tcells.cm.tn.val else exp.bulk.cm.tn.val
        resp.data <- resp.cm.tn.val
    } else if (ds %>% tolower == "brightness"){
        exp.data  <- if (use.ctp == "T-cells") exp.tcells.cm.bn else exp.bulk.cm.bn
        resp.data <- resp.cm.bn
    }
    
    ## perform enrichment-based prediction.
    exp.data <- exp.data %>% (function(df){
        df %>% apply(MARGIN = 1, var) %>% (function(x) df[-which(x == 0), ])
    }) %>% as.matrix
    
    enrich.res <- gsva(expr = exp.data, gset.idx.list = sign.tcells, 
                       method = "gsva", kcdf = "Gaussian", min.sz = 5, max.sz = 500, 
                       ssgsea.norm = T, verbose = T) %>% t %>% as.data.frame
    
    enrich.data <- enrich.res %>% mutate(
        CD4 = CD4 %>% mmscale, 
        CD8 = CD8 %>% mmscale, 
        resp = resp.data %>% factor(levels = c(1, 0)) )
    
    ## compute performance.
    perf.data <- lineages %>% names %>% sapply(simplify = T, function(ctp){
        classifier.performance(data = enrich.data, label = "resp", pred = ctp)
    }) %>% as.data.frame
    
    list(pred = enrich.data, perf = perf.data)                                      # save predictions & performance data
})

fcat("\nprediction performance for {use.ctp}:");  print(
    data.all %>% sapply(simplify = F, function(data.ds) data.ds$perf %>% round(4)))


#### -------------------------------------------------------------------------------

## prepare data for fig s3-I.
## panel A + C: R vs. NR.
fig.dataS3AC <- data.all %>% sapply(
    simplify = F, function(data.ds) data.ds$pred) %>% do.call(rbind, .) %>% 
    mutate(resp = resp %>% `levels<-`(c("R", "NR")), 
           cohort = rownames(.) %>% keep.first.str %>% factor(levels = datasets) %>% 
               `levels<-`(datasets %>% gsub(
                   pattern = " + ", replacement = " +\n", fixed = T) %>% 
                       gsub(pattern = "PBCP", replacement = "PBCP   ")))

fig.statS3AC <- lineages %>% names %>% sapply(simplify = F, function(ctp){
    fig.dataS3AC %>% group_by(cohort) %>% 
        wilcox_test(glue("{ctp} ~ resp") %>% as.formula, alternative = "g", 
                    p.adjust.method = "fdr") %>% add_significance("p") %>% 
        add_x_position(x = "cohort", group = "resp", dodge = 0.75) %>% 
        mutate(y.position = 1.05) 
})
    

## panel B + D: performance metrics.
fig.dataS3BD <- data.all %>% sapply(simplify = F, function(data.ds) 
    data.ds$perf %>% rownames_to_column("metric")) %>% do.call(rbind, .) %>% 
    mutate(metric = metric %>% factor(levels = c("AUC", "AP")), 
           cohort = rownames(.) %>% keep.first.str %>% factor(levels = datasets) %>% 
               `levels<-`(datasets %>% gsub(
                   pattern = " + ", replacement = " +\n", fixed = T) %>% 
                       gsub(pattern = "PBCP", replacement = "PBCP   ")))
    

#### -------------------------------------------------------------------------------

## make CD4/CD8 enrichment performance plots - fig s3-I.
font.name <- "sans"
font.size <- round(c("tick" = 56, "label" = 60, "title" = 80, "plabel" = 96) / 4)   # set denominator to 1 when saving the plot
plt.clrs <- c("R" = "#DC91AD", "NR" = "#EFCC74", "base" = "#000000")
dot.size <- 8 / 4                                                                   # set denominator to 1 when saving the plot
ln.size  <- 2 / 2                                                                   # set denominator to 1 when saving the plot

set.yticks <- scale_y_continuous(
    breaks = seq(0, 1, by = 0.2), expand = c(0.01, 0.01), limits = c(0, 1.12))
set.clrs  <- scale_fill_manual(values = plt.clrs[c("R", "NR")] %>% `names<-`(NULL)) # need to remove names for barplot
plt.theme <- theme(
    panel.grid = element_blank(), 
    axis.line = element_line(color = plt.clrs["base"], linewidth = ln.size), 
    axis.ticks = element_line(linewidth = ln.size, color = plt.clrs["base"]), 
    axis.ticks.length = unit(ln.size / 4, "cm"), 
    axis.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    legend.title = element_text(face = "bold", size = font.size["label"], 
                                color = plt.clrs["base"]), 
    legend.title.align = 0.45, 
    legend.key.size = unit(4, "line"), 
    legend.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    legend.text.align = 0)


fig.plotsS3 <- list()                                                               # list of all plots

fig.plotsS3[c("A", "C")] <- lineages %>% names %>% sapply(simplify = F, function(ctp){
    ggplot(data = fig.dataS3AC) + geom_boxplot(
        mapping = aes(x = cohort, y = .data[[ctp]], fill = resp), 
        linewidth = ln.size, fatten = 0.8, outlier.size = dot.size) + 
        xlab("") + ylab(latex2exp::TeX(glue("{ctp}$^+$  "), bold = T)) + 
        labs(fill = "Response    ") + 
        stat_pvalue_manual(fig.statS3AC[[ctp]], label = "p.signif", 
                           bracket.size = ln.size, vjust = -0.15, 
                           label.size = font.size["tick"] / 3, 
                           color = plt.clrs["base"]) + 
        theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
        set.clrs + set.yticks + plt.theme + theme(
            legend.position = "left", 
            axis.text.x = element_text(angle = 40, vjust = 1, hjust = 1, 
                                       size = font.size["tick"], 
                                       color = plt.clrs["base"]), 
            axis.title.y = element_text(angle = 0, vjust = 0.5, face = "bold", 
                                        size = font.size["title"], 
                                        color = plt.clrs["base"]))
})


fig.plotsS3[c("B", "D")] <- lineages %>% names %>% sapply(simplify = F, function(ctp){
    ggplot(data = fig.dataS3BD, 
           mapping = aes(x = cohort, y = .data[[ctp]], fill = metric)) + 
        geom_bar(stat = "identity", position = "dodge", width = 0.6, 
                 linewidth = ln.size, color = plt.clrs["base"]) + 
        geom_text(mapping = aes(
            label = .data[[ctp]] %>% sapply(function(x) sprintf("%0.2f", x))), 
            position = position_dodge(width = 0.6), hjust = 0.5, vjust = -0.35, 
            size = font.size["tick"] / 3, color = plt.clrs["base"]) + 
        xlab("") + ylab("") + labs(fill = "Performance") + 
        theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
        set.clrs + set.yticks + plt.theme + theme(
            axis.text.x = element_text(angle = 40, vjust = 1, hjust = 1, 
                                       size = font.size["tick"], 
                                       color = plt.clrs["base"]))
})


fig.plotsS3[["AC"]] <- ggarrange(
    fig.plotsS3$A + theme(axis.text.x = element_blank()), fig.plotsS3$C, 
    nrow = 2, ncol = 1, heights = c(0.72, 1.0), legend = "left", common.legend = T, 
    labels = c("A", "C"), label.x = 0.20, label.y = 1.06, font.label = list(
        family = font.name, size = font.size["plabel"], color = plt.clrs["base"])) 

fig.plotsS3[["BD"]] <- ggarrange(
    fig.plotsS3$B + theme(axis.text.x = element_blank(), axis.text.y = element_blank()), 
    fig.plotsS3$D + theme(axis.text.y = element_blank()), 
    nrow = 2, ncol = 1, heights = c(0.72, 1.0), legend = "right", common.legend = T, 
    labels = c("B", "D"), label.y = 1.06, font.label = list(
        family = font.name, size = font.size["plabel"], color = plt.clrs["base"])) 


fig.plotsS3[["final"]] <- ggarrange(
    plotlist = fig.plotsS3[c("AC", "BD")], nrow = 1, ncol = 2) + 
    theme(plot.margin = margin(l = -0.1, r = -0.1, t = -0.1, b = -0.1, unit = "cm"))

print(fig.plotsS3$final)


## save figure.
svdat <- F                                                                          # set T to save figure 
if (svdat){
    datestamp  <- DateTime()                                                        # datestamp for analysis
    fig.path   <- "../data/plots/"
    fig.fileS3 <- glue("all_chemo_CD4_CD8_performance_{use.ctp}_v2_{datestamp}.pdf")
    
    pdf(file = paste0(fig.path, fig.fileS3), height = 24, width = 48)
    print(fig.plotsS3$final)
    
    dev.off()
}

