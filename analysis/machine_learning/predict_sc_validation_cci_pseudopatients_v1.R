#### -------------------------------------------------------------------------------
#### created on 14 sep 2023, 04:19pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(PRROC)
library(rstatix)
library(ggpubr)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))

resp.from.cci <- function(ccis, patients){
    X <- patients[ccis$CCI, ] %>% t
    w <- ifelse(ccis$Direction > 0, 1, -1)                                          # weight CCI by +1 / -1
    y.hat <- X %*% w %>% drop %>% (function(y) (y - min(y)) / diff(range(y)))
}

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## load data.
data.path <- "../data/TransNEO/transneo_analysis/mdl_data/"
data.file <- c("SC_data_for_top_170CCIs_SRD_18Sep2023.RDS", 
               "DhrubaBRCA_TPM_pseudopat_scmap_seed20892_rslurm_200pat_30ds_version2.rds")

## list of top CCIs.
cci.data <- readRDS(paste0(data.path, data.file[1]))
cci.list <- cci.data$cci.list %>% mutate(
    "CCI" = paste(LigandCell, ReceptorCell, LigandGene, ReceptorGene, sep = "_"), 
    "cci.annot" = paste(paste(LigandCell, ReceptorCell, sep = " - "), 
                        paste(LigandGene, ReceptorGene, sep = " - "), sep = " : : "), 
    "Direction.adj" = Direction %>% abs %>% p.adjust(method = "fdr") %>% 
        `*`(ifelse(Direction > 0, 1, -1))                                           # run FDR correction
) %>% `rownames<-`(.$CCI)


## patient profile: rows: cell-cell-L-R quadruplet, cols: patients.
patient.data    <- readRDS(paste0(data.path, data.file[2]))
patient.profile <- patient.data$pval %>% (function(df) df[!duplicated(df$int), ]) %>% 
    `rownames<-`(NULL) %>% column_to_rownames("int")


## calculate patient binary score from empirical p-values: 
## Present = (LR == 1), Absent = (LR == 0).
pval.cut <- 0.05
cci.profile <- patient.profile %>% apply(MARGIN = 2, function(x){
    ifelse(x < pval.cut, 1, 0)
})

response <- patient.profile %>% colnames %>% sapply(function(x){
    ifelse(substr(x, start = 1, stop = 1) == "R", 1, 0)
})


## predict response from CCIs.
pval.sig   <- 0.05                                                                  # directionality significance cut-off
cci.use    <- cci.list %>% filter(abs(Direction.adj) <= pval.sig)                   # use CCIs with significant directionalities
prediction <- resp.from.cci(patients = cci.profile, ccis = cci.use)
res.roc    <- roc.curve(weights.class0 = response, scores.class0 = prediction, 
                        curve = T)
fcat("\nperformance summary:
     top CCIs from bulk (p-value ≤ {pval.sig}): m = {cci.use %>% nrow}
     test on: pseudo-patient profile (n = {cci.profile %>% nrow})
     AUC = {res.roc$auc %>% round(2)}")


#### -------------------------------------------------------------------------------

## prepare data for fig s4-II.
fig.dataS4E <- data.frame(label = response %>% factor(levels = c(1, 0)) %>% 
                              `levels<-`(c("R", "NR")), 
                          score = prediction)
fig.statS4E <- fig.dataS4E %>% wilcox_test(
    score ~ label, alternative = "greater", p.adjust.method = "fdr") %>% 
    add_significance("p") %>% add_x_position(x = "label", group = "label") %>% 
    mutate(y.position = 1.06)

fig.dataS4F <- res.roc$curve %>% `colnames<-`(c("FPR", "TPR", "Th")) %>% 
    as.data.frame
fig.statS4F <- res.roc$auc


## make enrichment plot - fig s4-II.
font.name <- "sans"
font.size <- round(c("tick" = 56, "label" = 60, "title" = 72, "plabel" = 96) / 4)   # set denominator to 1 when saving the plot
plt.clrs <- c("R" = "#DC91AD", "NR" = "#EFCC74", "base" = "#000000")
dot.size <- c("out" = 8, "pt" = 12) / 3                                             # set denominator to 1 when saving the plot
ln.size  <- 2 / 2                                                                   # set denominator to 1 when saving the plot

set.xticks <- scale_x_continuous(
    breaks = seq(0, 1, by = 0.2), expand = c(0.01, 0.01), limits = c(0, 1.02))
set.yticks <- scale_y_continuous(
    breaks = seq(0, 1, by = 0.2), expand = c(0.01, 0.01), limits = c(0, 1.12))
set.clrs  <- scale_fill_manual(values = plt.clrs[c("R", "NR")])
plt.theme <- theme(
    panel.grid = element_blank(), 
    axis.line = element_line(color = plt.clrs["base"], linewidth = ln.size), 
    axis.ticks = element_line(linewidth = ln.size, color = plt.clrs["base"]), 
    axis.ticks.length = unit(ln.size / 4, "cm"), 
    axis.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    plot.title = element_text(hjust = 0.5, face = "bold", size = font.size["title"], 
                              color = plt.clrs["base"]), 
    legend.title = element_text(face = "bold", size = font.size["label"], 
                                color = plt.clrs["base"]), 
    legend.title.align = 0.45, 
    legend.key.size = unit(4, "line"), 
    legend.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    legend.text.align = 0)


fig.plotS4 <- list()                                                                # list of all plots

fig.plotS4[["E"]] <- ggplot(data = fig.dataS4E, mapping = aes(x = label, y = score)) + 
    geom_boxplot(mapping = aes(fill = label), color = plt.clrs["base"], 
                 linewidth = ln.size, fatten = 0.8, outlier.size = dot.size["out"], 
                 show.legend = F) + xlab("") + ylab("") + 
    ggtitle("SC-TNBC response") + stat_pvalue_manual(
        fig.statS4E, label = "p.signif", bracket.size = ln.size, vjust = 0, 
        label.size = font.size["tick"] / 2, color = plt.clrs["base"]) + 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    set.clrs + set.yticks + plt.theme + theme(
        axis.text.x = element_text(size = font.size["label"], 
                                   color = plt.clrs["base"]))


fig.plotS4[["F"]] <- ggplot(data = fig.dataS4F, mapping = aes(x = FPR, y = TPR)) + 
    geom_line(linewidth = ln.size, color = plt.clrs["R"]) + 
    geom_point(size = dot.size["pt"], color = plt.clrs["R"]) + 
    geom_abline(intercept = 0, slope = 1, linetype = "dotted", linewidth = ln.size, 
                color = plt.clrs["base"]) + 
    xlab("1 - Specificity") + ylab("Sensitivity") + ggtitle("SC-TNBC performance") + 
    annotate(geom = "text", x = 0.75, y = 0.65, 
             label = sprintf("AUC = %0.2f", fig.statS4F), 
             size = font.size["tick"] / 2, color = plt.clrs["base"], 
             fontface = "bold") + 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    set.xticks + set.yticks + plt.theme


fig.plotS4[["final"]] <- ggarrange(
    fig.plotS4$E, fig.plotS4$`F`, nrow = 1, ncol = 2, labels = c("E", "F"), 
    font.label = list(family = font.name, size = font.size["plabel"], 
                      color = plt.clrs["base"]))

print(fig.plotS4$final)


## save figure.
svdat <- F                                                                          # set T to save figure 
if (svdat){
    datestamp  <- DateTime()                                                        # datestamp for analysis
    fig.path   <- "../data/TransNEO/transneo_analysis/plots/"
    fig.fileS4 <- glue("tnbc_sc_validation_cci_pseudopatients{pval.cut}_p{pval.sig}_{datestamp}.pdf")
    
    pdf(file = paste0(fig.path, fig.fileS4), height = 16, width = 48)
    print(fig.plotS4$final)
    
    dev.off()
}

