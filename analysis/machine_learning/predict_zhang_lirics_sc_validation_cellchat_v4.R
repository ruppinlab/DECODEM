#### -------------------------------------------------------------------------------
#### created on 07 nov 2025, 04:30pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

if (Sys.info()["sysname"] == "Darwin"){                                             # mac
    .wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/"
    .mapth. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/r/miscellaneous.R"
} else if (Sys.info()["sysname"] == "Linux"){                                       # biowulf
    .wpath. <- "/data/Lab_ruppin/projects/TME_contribution_project/analysis/LIRICS_and_SOCIAL/CellChat/"
    .mpath. <- "/home/dhrubas2/vivid/miscellaneous.R"
}

setwd(.wpath.)
source(.mapth.)

library(CellChat)
library(PRROC)
library(rstatix)
library(ggpubr)


## functions.
fcat <- function(..., end = "\n", start = "") cat(start, paste0(glue(...), end))    # f-string print akin to python

predict.from.ccis <- function(data, ccis, weights){                                 # predict response from given CCI profile & weights
    if (ccis %>% missing) ccis <- data %>% colnames
    if (weights %>% missing) weights <- rep(1, ccis %>% length)
    
    X     <- data[ccis] %>% as.matrix
    w     <- weights / length(weights)
    y.hat <- X %*% w
    y.hat
}


cat("\014")                                                                         # clears console


### --------------------------------------------------------------------------------

## read data.
data.path <- "../../data/SC_data/ZhangTNBC2021/validation/"
data.file <- "SC_data_for_top_170CCIs_SRD_18Sep2023.RDS"

fcat("loading & preparing data...");    .dt <- Sys.time()

sc.data <- readRDS(paste0(data.path, data.file))
sc.data$cci.list <- sc.data$cci.list %>% mutate(
    CCI = paste(LigandCell, ReceptorCell, LigandGene, ReceptorGene, sep = "_"),
    Direction.adj = p.adjust(p = abs(Direction), method = "fdr") * 
        ifelse(Direction > 0, yes = 1, no = -1))

sc.exp  <- sc.data$sc.exp.all %>% log1p %>% as.matrix
sc.meta <- sc.data$sc.annot %>% column_to_rownames(var = "Cell.id")

.dt <- Sys.time() - .dt;    fcat("done! elapsed time = {.dt %>% as.numeric(units = 'secs') %>% round(2)} secs.")

fcat("patients = {sc.meta$Sample.id %>% unique %>% length}, cells = {sc.meta %>% nrow}", 
     start = "\n")


#### -------------------------------------------------------------------------------

## CCI database: L-R pairs from top CCIs from bulk.
LR.db  <- sc.data$cci.list %>% 
    select(ligand = LigandGene, receptor = ReceptorGene) %>% 
    mutate(interaction_name = paste(ligand, receptor, sep = "_")) %>% 
    unique

db.new <- updateCellChatDB(db = LR.db, gene_info = NULL, merged = F, 
                           species_target = "human")


#### -------------------------------------------------------------------------------

## prepare for CCI inference.
## run CellChat per sample to infer sample-specific CCI profile.

future::plan("multisession", workers = 2)
options(future.globals.maxSize = Inf)

min.cells <- 5
pval.cut  <- 0.05

cci.net   <- sc.meta$Sample.id %>% unique %>% sapply(simplify = F, function(smpl){
    ## prepare data.
    fcat("sample = ", smpl, start = "\n")                                           # glue can't interpret apply-only variables
    meta    <- sc.meta %>% 
        mutate(samples = Sample.id %>% as.factor) %>% 
        filter(samples == smpl)
    exp     <- sc.exp[, meta %>% rownames]
    
    ## make cellchat object with custom L-R database.
    clct    <- createCellChat(object = exp, meta = meta, group.by = "Cell.type")
    clct@DB <- db.new
    clct    <- subsetData(clct)
    clct    <- identifyOverExpressedGenes(clct, min.cells = min.cells, 
                                          thresh.pc = 0.1, thresh.fc = 0.1, 
                                          thresh.p = pval.cut)
    clct    <- identifyOverExpressedInteractions(clct)
    
    ## infer CCI probabilities.
    .dt     <- Sys.time()
    
    clct    <- computeCommunProb(clct, type = "triMean")                            # tried median- doesn't work
    clct    <- filterCommunication(clct, min.cells = min.cells)                     # include results only if 5+ cells exist 
    net     <- subsetCommunication(clct, thresh = pval.cut) %>%
        mutate(CCI = paste(source, target, interaction_name, sep = "_"))            # outputs significant CCIs only
    
    .dt     <- Sys.time() - .dt;    fcat("elapsed time = ", 
                                         .dt %>% as.numeric(units = 'secs') %>% round(2), 
                                         " secs.")
    net
})


#### -------------------------------------------------------------------------------

## format into CCI matrix.
bin.ccis <- T
prob.cut <- 0.2                                                                     # activation cut-off

ccis.all <- cci.net %>% sapply(function(net) net$CCI) %>% reduce(.f = union)        # total set of significant CCIs across samples
cci.mat  <- cci.net %>% sapply(function(net){
    mat <- rep(0, ccis.all %>% length) %>% `names<-`(ccis.all)
    mat[net$CCI] <- net$prob / max(net$prob)                                        # scaled interaction score
    if (bin.ccis)
        mat <- ifelse(mat >= prob.cut, yes = 1, no = 0)                             # binarize CCI matrix
    mat
}) %>% t %>% as.data.frame

fcat("total identified CCIs = {cci.mat %>% ncol}")


## compute response prediction.
fcat("predicting response based on CCIs...", start = "\n");    .dt <- Sys.time()

sc.resp <- sc.meta %>% 
    select(Sample.id, Response) %>% 
    unique %>% 
    (function(df) df$Response %>% `names<-`(df$Sample.id))

use.cmn <- T
if (use.cmn){
    ccis.use <- intersect(ccis.all, sc.data$cci.list$CCI)                           # overlap with bulk CCIs
    wgts.cci <- sc.data$cci.list %>% (function(df){
        ifelse(df$Direction.adj > 0, yes = 1, no = -1) %>% 
            `names<-`(df$CCI) %>% 
            .[ccis.use]
    })
} else {                                                                            # all significant CCIs
    ccis.use <- ccis.all
    wgts.cci <- rep(1, length(ccis.use))
}

fcat("using m = {ccis.use %>% length} top CCIs...")
sc.pred <- predict.from.ccis(data = cci.mat, ccis = ccis.use, weights = wgts.cci)

.dt <- Sys.time() - .dt;    fcat("prediction done! elapsed time = {.dt %>% as.numeric(units = 'secs') %>% round(2)} secs.")


## evaluate scores.
sc.pmat <- data.frame(label = sc.resp, score = sc.pred)
sc.roc  <- roc.curve(weights.class0 = sc.pmat$label, 
                     scores.class0 = sc.pmat$score, curve = T)
sc.stat <- sc.pmat %>% 
    mutate(label = label %>% factor(levels = c(1, 0))) %>% 
    wilcox_test(score ~ label, alternative = "greater", p.adjust.method = "fdr")


fcat("
validation performance for treatment = chemotherapy:
using cell types: {sc.meta$Cell.type %>% unique %>% paste(collapse = ', ')}
cohort = Zhang et al. (n = {sc.pmat %>% nrow}, R:NR = {sc.resp %>% sum}:{sc.resp %>% `!` %>% sum})
#CCIs  = total: {ccis.all %>% length}, overlap: {ccis.use %>% length}
AUC    = {sc.roc$auc %>% round(3)}, P = {sc.stat$p %>% round(3)}
")


#### -------------------------------------------------------------------------------

## prepare plot data for fig 4G-H.
## panel G: R vs. NR.
fig.data4G <- sc.pmat %>% mutate(
    label = label %>% factor(levels = c(1, 0)) %>% `levels<-`(c("R", "NR")))

fig.stat4G <- sc.stat %>% 
    add_significance("p") %>% 
    add_x_position(x = "label", group = "label") %>% 
    mutate(y.position = 1.01, p.lbl = glue("P = {p %>% round(3)}"))


## panel H: ROC curve.
fig.data4H <- sc.roc$curve %>% 
    `colnames<-`(c("FPR", "TPR", "Th")) %>% 
    as.data.frame

fig.stat4H <- sc.roc$auc %>% round(2)


#### -------------------------------------------------------------------------------

## make fig 4G-H: SC CCI performance plot.
## plot parameters.
font.name <- "sans"
font.size <- c("tick" = 20, "label" = 24, "title" = 32, "plabel" = 60) / 1.5
plt.clrs  <- c("R" = "#E08DAC", "NR" = "#7595D0", "score" = "#B075D0", 
               "box" = "#A9A9A9", "base" = "#000000")
dot.size  <- c("out" = 4, "pt" = 6) / 1.25
ln.size   <- c("main" = 1, "base" = 0.75)

plt.theme <- theme(
    panel.grid = element_blank(), 
    axis.line = element_line(color = plt.clrs["base"], linewidth = ln.size["main"]), 
    axis.ticks = element_line(linewidth = ln.size["base"], color = plt.clrs["base"]), 
    axis.ticks.length = unit(ln.size["main"] / 2, "cm"), 
    axis.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    plot.title = element_text(hjust = 0.5, vjust = 1.04, face = "bold", 
                              size = font.size["title"], color = plt.clrs["base"]), 
    legend.title = element_text(hjust = 0.5, face = "bold", size = font.size["label"], 
                                color = plt.clrs["base"]), 
    legend.key.size = unit(4, "line"), 
    legend.text = element_text(hjust = 0, size = font.size["tick"], 
                               color = plt.clrs["base"]))


fig.ttls4_II <- c(glue("Prediction score: Zhang et al. (n = {fig.data4G %>% nrow})"), 
                  glue("ROC curve: Zhang et al. (n = {fig.data4G %>% nrow})"))
fig.lims4_II <- c(0.2, 0.05)
fig.fill     <- T


## make figures.
fig.plot4_II <- list()                                                              # list of all plots

## panel G: R vs. NR violin plot.
fig.plot4_II[["G"]] <- ggplot(
    data = fig.data4G, mapping = aes(x = label, y = score)) + 
    geom_violin(mapping = aes(fill = label), color = plt.clrs["base"], 
                stat = "ydensity", scale = "area", bw = "bcv", trim = F, 
                na.rm = T, width = 0.6, linewidth = ln.size["main"], 
                show.legend = F) + 
    geom_boxplot(width = 0.1, fill = plt.clrs["box"], linewidth = ln.size["base"], 
                 median.linewidth = 0.8, outlier.size = dot.size["out"]) +
    xlab("") + ylab("") + ggtitle(fig.ttls4_II[1]) + 
    stat_pvalue_manual(fig.stat4G, label = "p.lbl", remove.bracket = T, vjust = -3, 
                       hjust = 1.5, label.size = font.size["tick"] / 2, 
                       color = plt.clrs["base"]) + 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    scale_fill_manual(values = plt.clrs[c("R", "NR")]) + 
    scale_y_continuous(breaks = seq(-0.0, 1.0, by = 0.2), expand = c(0.01, 0.01), 
                       limits = c(0 - fig.lims4_II[1], 1 + fig.lims4_II[1])) +
    plt.theme + theme(axis.text.x = element_text(size = font.size["label"], 
                                                 color = plt.clrs["base"]))

# print(fig.plot4_II$G)

## panel H: ROC curve with AUC.
fig.plot4_II[["H"]] <- ggplot(data = fig.data4H, mapping = aes(x = FPR, y = TPR)) + 
    geom_line(linetype = "solid", linewidth = ln.size["main"], 
              color = plt.clrs["score"], show.legend = T) + 
    geom_point(shape = "circle", size = dot.size["pt"], color = plt.clrs["score"]) + 
    geom_line(mapping = aes(x = FPR, y = FPR), linetype = "longdash", 
              linewidth = ln.size["base"], color = plt.clrs["base"], 
              show.legend = T) + 
    xlab(latex2exp::TeX("1 $-$ Specificity")) + ylab("Sensitivity") + 
    ggtitle(fig.ttls4_II[2]) + 
    annotate(geom = "text", x = 0.75, y = 0.50, 
             label = sprintf("AUC = %0.2f", fig.stat4H), 
             size = font.size["tick"] / 2, color = plt.clrs["base"], 
             fontface = "bold") + 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    scale_x_continuous(breaks = seq(0, 1., by = 0.2), expand = c(0.01, 0.01), 
                       limits = c(0 - fig.lims4_II[2], 1 + fig.lims4_II[2])) + 
    scale_y_continuous(breaks = seq(0, 1., by = 0.2), expand = c(0.01, 0.01), 
                       limits = c(0 - fig.lims4_II[2], 1 + fig.lims4_II[2])) + 
    plt.theme

if (fig.fill){                                                                      # fill area under ROC curve
    fig.plot4_II[["H"]] <- fig.plot4_II[["H"]] + geom_ribbon(
        mapping = aes(x = FPR, ymin = FPR, ymax = TPR), 
        fill = plt.clrs["score"], alpha = 0.3)
}

# print(fig.plot4_II$H)


## final plot.
fig.plot4_II[["final"]] <- ggarrange(
    fig.plot4_II$G, fig.plot4_II$H, nrow = 1, ncol = 2, labels = c("G", "H"), 
    label.y = 1.04, font.label = list(family = font.name, 
                                      size = font.size["plabel"], 
                                      color = plt.clrs["base"]))

cat("\014")                                                                         # clears console
print(fig.plot4_II$final)


#### -------------------------------------------------------------------------------

## save plot.
svdat <- F                                                                          # set as T to save figure

if (svdat){
    fig.path     <- "../../data/TransNEO/transneo_analysis/plots/final_plots7/"
    fig.file4_II <- glue("tnbc_sc_validation_cci_cellchat_act{prob.cut}_p{pval.cut}.pdf")
    
    ggsave(path = fig.path, filename = fig.file4_II, plot = fig.plot4_II$final, 
           device = "pdf", dpi = 600, height = 8, width = 18, units = "in")
    print(fig.plot4_II$final)
    
    fcat(fig.file4_II)
    dev.off()
}

