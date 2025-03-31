setwd("/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/")
source("/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/r/miscellaneous.R")

library(PRROC)
library(rstatix)
library(ggpubr)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                            # f-string print akin to python

mmscale <- function(x) (x - x %>% min) / (x %>% range %>% diff)                     # rescale data in [0, 1]

resp.from.cci <- function(ccis, patients){
    X <- patients[ccis$CCI, ] %>% t
    w <- ifelse(ccis$Direction > 0, 1, -1)                                          # weight CCI by +1 / -1
    y.hat <- X %*% w %>% drop %>% mmscale
    y.hat
}

classifier.performance <- function(true, pred){
    list("AUC" = roc.curve(weights.class0 = true, scores.class0 = pred)$auc, 
         "AP"  = pr.curve(weights.class0 = true, scores.class0 = pred)$auc.integral)
}

get.roc.curve <- function(true, pred, th.list){
    if (th.list %>% missing){
        th.list <- seq(0, 1, by = 0.02)
    }
    
    curve.data <- th.list %>% sapply(function(th){
        pred.th <- ifelse(pred >= th, 1, 0)
        tp      <- sum((true == 1) & (pred.th == 1))
        fp      <- sum((true == 0) & (pred.th == 1))
        fn      <- sum((true == 1) & (pred.th == 0))
        tn      <- sum((true == 0) & (pred.th == 0))
        c("Th" = th, "TPR" = tp / (tp + fn), "FPR" = fp / (fp + tn))
    })
    
    curve.data %>% as.matrix %>% t %>% as.data.frame
}

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## load data.
data.path <- "../../data/TransNEO/transneo_analysis/mdl_data/"
data.file <- c("SC_data_for_top_170CCIs_SRD_18Sep2023.RDS", 
               "DhrubaBRCA_TPM_pseudopat_scmap_seed20892_rslurm_200pat_30ds_version3.rds")

## list of top CCIs.
cci.data <- readRDS(paste0(data.path, data.file[1]))
cci.list <- cci.data$cci.list %>% mutate(
    "CCI" = paste(LigandCell, ReceptorCell, LigandGene, ReceptorGene, sep = "_"), 
    "Direction.adj" = Direction %>% abs %>% p.adjust(method = "fdr") %>% 
        `*`(ifelse(Direction > 0, 1, -1))                                           # run FDR correction
) %>% select(CCI, MDI, Direction.adj, CCIannot)

cci.data %>% rm                                                                     # release memory


## patient profile: rows: cell-cell-L-R quadruplet, cols: patients.
## generate CCI activation profile for patients from empirical p-values.
## present: LR == 1, absent: LR == 0.
pval.cut <- 0.05                                                                    # p-value significance cut-off

patient.data    <- readRDS(paste0(data.path, data.file[2]))
patient.profile <- patient.data$pval %>% (function(pval.mat){
    cci.act  <- pval.mat[, -1] %>% as.matrix %>% `rownames<-`(pval.mat$int) %>%
        apply(MARGIN = 2, function(p){ ifelse(p <= pval.cut, 1, 0) })               # binary activation status

    pat.prof <- pval.mat$int %>% unique %>% sapply(simplify = T, function(cci){
        cci.act[cci, , drop = F] %>% apply(MARGIN = 2, median) %>% round            # mean activation status
    }) %>% as.data.frame

    pat.prof$response <- pat.prof %>% rownames %>% sapply(function(pat){            # extract response labels
        ifelse(substr(pat, start = 1, stop = 1) == "R", 1, 0)
    })
    
    pat.prof %>% t
})

patient.ccis <- patient.profile[cci.list$CCI, ]                                     # use only the top CCIs
patient.resp <- patient.profile["response", ]

patient.data %>% rm                                                                 # release memory


## predict response from CCIs.
cci.use      <- cci.list %>% filter(abs(Direction.adj) <= pval.cut)                 # use CCIs with significant directionalities
patient.pred <- resp.from.cci(patients = patient.ccis, ccis = cci.use)
cci.scores   <- classifier.performance(true = patient.resp, pred = patient.pred)
fcat("\nperformance summary:
     top CCIs from bulk (p-value ≤ {pval.cut}): m = {cci.use %>% nrow}
     test on: pseudo-patient profile (n = {patient.ccis %>% ncol})
     performance: AUC = {cci.scores$AUC %>% round(4)}, AP = {cci.scores$AP %>% round(4)}")


#### -------------------------------------------------------------------------------

## prepare plot data.
fig.data4E <- data.frame(
    label = patient.resp %>% factor(levels = c(1, 0)) %>% `levels<-`(c("R", "NR")), 
    score = patient.pred)
fig.stat4E <- fig.data4E %>% wilcox_test(
    score ~ label, alternative = "greater", p.adjust.method = "fdr") %>% 
    add_significance("p") %>% 
    add_x_position(x = "label", group = "label") %>% 
    mutate(y.position = 1.06)

fig.data4F <- get.roc.curve(true = patient.resp, pred = patient.pred)
fig.stat4F <- cci.scores$AUC %>% round(2)


## make performance plot.
font.name <- "sans"
font.size <- c("tick" = 20, "label" = 24, "title" = 32, "plabel" = 60)
plt.clrs <- c("R" = "#E08DAC", "NR" = "#7595D0", "score" = "#B075D0", 
              "box" = "#A9A9A9", "base" = "#000000")
dot.size <- c("out" = 4, "pt" = 6)
ln.size  <- c("main" = 1, "base" = 0.75)

plt.theme <- theme(
    panel.grid = element_blank(), 
    axis.line = element_line(color = plt.clrs["base"], linewidth = ln.size["main"]), 
    axis.ticks = element_line(linewidth = ln.size["base"], color = plt.clrs["base"]), 
    axis.ticks.length = unit(ln.size["main"], "cm"), 
    axis.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    plot.title = element_text(hjust = 0.5, vjust = 1.04, face = "bold", 
                              size = font.size["title"], color = plt.clrs["base"]), 
    legend.title = element_text(hjust = 0.5, face = "bold", size = font.size["label"], 
                                color = plt.clrs["base"]), 
    legend.key.size = unit(4, "line"), 
    legend.text = element_text(hjust = 0, size = font.size["tick"], 
                               color = plt.clrs["base"]))


fig.ttls4_II <- c(glue("Prediction score: SC-TNBC (n = {fig.data4E %>% nrow})"), 
                  glue("Model performance: SC-TNBC (n = {fig.data4E %>% nrow})"))
fig.lims4_II <- c(0.25, 0.05)
fig.fill     <- T

fig.plot4_II <- list()

## generate plot.
fig.plot4_II[["E"]] <- ggplot(
    data = fig.data4E, mapping = aes(x = label, y = score)) + 
    geom_violin(mapping = aes(fill = label), color = plt.clrs["base"], 
                stat = "ydensity", scale = "area", bw = "bcv", trim = F, 
                na.rm = T, width = 0.7, linewidth = ln.size["main"], 
                show.legend = F) + 
    geom_boxplot(width = 0.15, fill = plt.clrs["box"], linewidth = ln.size["base"], 
                 fatten = 0.8, outlier.size = dot.size["out"]) +
    xlab("") + ylab("") + ggtitle(fig.ttls4_II[1]) + stat_pvalue_manual(
        fig.stat4E, label = "p.signif", bracket.size = 0, vjust = -2, 
        label.size = font.size["tick"] / 2, color = plt.clrs["base"]) + 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    scale_fill_manual(values = plt.clrs[c("R", "NR")]) + 
    scale_y_continuous(
        breaks = seq(-0.2, 1.2, by = 0.2), expand = c(0.01, 0.01), 
        limits = c(0 - fig.lims4_II[1], 1 + fig.lims4_II[1])) + 
    plt.theme + theme(axis.text.x = element_text(size = font.size["label"], 
                                                 color = plt.clrs["base"]))

# print(fig.plot4_II$E)

fig.plot4_II[["F"]] <- ggplot(data = fig.data4F, mapping = aes(x = FPR, y = TPR)) + 
    geom_line(linetype = "solid", linewidth = ln.size["main"], 
              color = plt.clrs["score"], show.legend = T) + 
    geom_point(shape = "circle", size = dot.size["pt"], color = plt.clrs["score"]) + 
    geom_line(mapping = aes(x = FPR, y = FPR), linetype = "longdash", 
              linewidth = ln.size["base"], color = plt.clrs["base"], 
              show.legend = T) + 
    xlab(latex2exp::TeX("1 $-$ Specificity")) + ylab("Sensitivity") + 
    ggtitle(fig.ttls4_II[2]) + 
    annotate(geom = "text", x = 0.85, y = 0.60, 
             label = sprintf("AUC = %0.2f", fig.stat4F), 
             size = font.size["tick"] / 2, color = plt.clrs["base"], 
             fontface = "bold") + 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    scale_x_continuous(
        breaks = seq(-0.2, 1.2, by = 0.2), expand = c(0.01, 0.01), 
        limits = c(0 - fig.lims4_II[2], 1 + fig.lims4_II[2])) + 
    scale_y_continuous(
        breaks = seq(-0.2, 1.2, by = 0.2), expand = c(0.01, 0.01), 
        limits = c(0 - fig.lims4_II[2], 1 + fig.lims4_II[2])) + 
    plt.theme

if (fig.fill){
    fig.plot4_II[["F"]] <- fig.plot4_II[["F"]] + geom_ribbon(
        mapping = aes(x = FPR, ymin = FPR, ymax = TPR), 
        fill = plt.clrs["score"], alpha = 0.3)
}

# print(fig.plot4_II$F)

fig.plot4_II[["final"]] <- ggarrange(
    fig.plot4_II$E, fig.plot4_II$F, nrow = 1, ncol = 2, labels = c("E", "F"), 
    label.y = 1.04, font.label = list(family = font.name, 
                                      size = font.size["plabel"], 
                                      color = plt.clrs["base"]))

print(fig.plot4_II$final)


## save plot.
svdat <- F                                                                          # set as T to save figure
if (svdat){
    fig.path     <- "../../data/TransNEO/transneo_analysis/plots/final_plots6/"
    fig.file4_II <- glue("tnbc_sc_validation_cci_pseudopatients{pval.cut}_p{pval.cut}.pdf")
    
    pdf(file = paste0(fig.path, fig.file4_II), height = 8, width = 18)
    print(fig.plot4_II$final)
    
    dev.off()
}

