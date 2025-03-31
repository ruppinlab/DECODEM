setwd("/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/")
source("/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/r/miscellaneous.R")

library(GSVA)
library(PRROC)
library(rstatix)
library(ggpubr)
library(latex2exp)

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

mmscale <- function(x) (x - x %>% min) / (x %>% range %>% diff)                     # rescale data in [0, 1]

classifier.performance <- function(data, label, pred, method = "integral"){
    ## compute average precision (method = integral / davis.goodrich).
    if (data %>% missing){ true <- label;   pred <- pred }
    else { true <- data[[label]];   pred <- data[[pred]] }
    if (true %>% is.factor) true <- (true %>% levels %>% as.numeric)[true]
    if (pred %>% is.factor) pred <- (pred %>% levels %>% as.numeric)[pred]
    
    perf <- c("AUC" = roc.curve(scores.class0 = pred, weights.class0 = true)$auc, 
              "AP"  = pr.curve(scores.class0 = pred, weights.class0 = true)[[
                  glue("auc.{method}")]])
    perf
}

cat("\014")                                                                         # clears console


#### -------------------------------------------------------------------------------

## read sc DEGs & exp/resp data.
data.path <- c("../../data/celltype_signature/", 
               "../../data/TransNEO/CODEFACS_results/", 
               "../../data/TransNEO/use_data/", 
               "../../data/TransNEO/TransNEO_SammutShare/out_codefacs_tn_val_v2/", 
               "../../data/TransNEO/TransNEO_SammutShare/", 
               "../../data/BrighTNess/out_codefacs_brightness_v2/", 
               "../../data/BrighTNess/")

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
use.ctp  <- "T-cells"                                                               # "T-cells", "Bulk

data.all <- datasets %>% sapply(simplify = F, function(ds){
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
    
    exp.data <- exp.data %>% (function(df){
        df %>% apply(MARGIN = 1, var) %>% (function(x){
            idx <- which(x == 0)
            if (idx %>% length > 0) df[-which(x == 0), ]
            df })
    }) %>% as.matrix
    
    enrich <- gsva(gsvaParam(exprData = exp.data, geneSets = sign.tcells, 
                             kcdf = "Gaussian", minSize = 5, maxSize = 500), 
                   verbose = T) %>% t %>% as.data.frame
    
    enrich.data <- enrich %>% mutate(
        CD4 = CD4 %>% mmscale, CD8 = CD8 %>% mmscale, 
        resp = resp.data %>% factor(levels = c(1, 0)) )
    
    perf.data <- lineages %>% names %>% sapply(simplify = T, function(ctp){
        classifier.performance(data = enrich.data, label = "resp", pred = ctp)
    }) %>% as.data.frame
    
    list(pred = enrich.data, perf = perf.data)                                      # save predictions & performance data
})

fcat("\nprediction performance for T-cells:");  print(
    data.all %>% sapply(simplify = F, function(data.ds) data.ds$perf %>% round(4)))


#### -------------------------------------------------------------------------------

## prepare data for supp. fig. 4-I.
## panel A + C: R vs. NR.
n.datasets   <- data.all %>% names %>% sapply(function(ds){
    data.all[[ds]]$pred %>% nrow}) %>% data.frame(n = .) %>% 
    rownames_to_column(var = "dataset")
n.datasets$label = n.datasets %>% apply(MARGIN = 1, function(x){
    glue("{x[1]}\n(n = {x[2] %>% as.numeric})")})

fig.dataS4AC <- data.all %>% sapply(
    simplify = F, function(data.ds) data.ds$pred) %>% do.call(rbind, .) %>% 
    mutate(resp    = resp %>% `levels<-`(c("R", "NR")), 
           dataset = rownames(.) %>% keep.first.str %>% factor(levels = datasets) %>% 
               `levels<-`(n.datasets$label))

fig.statS4AC <- lineages %>% names %>% sapply(simplify = F, function(ctp){
    fig.dataS4AC %>% group_by(dataset) %>% 
        wilcox_test(glue("{ctp} ~ resp") %>% as.formula, alternative = "g", 
                    p.adjust.method = "fdr") %>% add_significance("p") %>% 
        add_x_position(x = "dataset", group = "resp", dodge = 0.75) %>% 
        mutate(y.position = 1.05) 
})
    

## panel B + D: performance metrics.
fig.dataS4BD <- data.all %>% sapply(simplify = F, function(data.ds) 
    data.ds$perf %>% rownames_to_column("metric")) %>% do.call(rbind, .) %>% 
    mutate(metric  = metric %>% factor(levels = c("AUC", "AP")), 
           dataset = rownames(.) %>% keep.first.str %>% factor(levels = datasets) %>% 
               `levels<-`(n.datasets$label))
    

#### -------------------------------------------------------------------------------

## plot functions: make paired half violin plot.
## make geom for combining with ggplot.
GeomSplitViolin <- ggplot2::ggproto(
    "GeomSplitViolin", ggplot2::GeomViolin, draw_group = function(
        self, data, ..., nudge = 0, draw_quantiles = NULL) {
        data <- data %>% transform(xminv = x - violinwidth * (x - xmin), 
                                   xmaxv = x + violinwidth * (xmax - x))
        grp  <- data[1, "group"]
        newdata <- plyr::arrange(
            transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), 
            if (grp %% 2 == 1) y else -y)
        newdata <- rbind(
            newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
        newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- 
            round(newdata[1, "x"])
        
        # now nudge them apart
        newdata$x <- ifelse(newdata$group %% 2 == 1, 
                            newdata$x - nudge, newdata$x + nudge)
        
        if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
            stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 1))
            # quantiles <- ggplot2:::create_quantile_segment_frame(
            #     data, draw_quantiles)
            quantiles <- create_quantile_segment_frame(
                data, draw_quantiles, split = TRUE, grp = grp)
            aesthetics <- data[rep(1, nrow(quantiles)), 
                               setdiff(names(data), c("x", "y")), drop = FALSE]
            aesthetics$alpha <- rep(1, nrow(quantiles))
            both <- cbind(quantiles, aesthetics)
            quantile_grob <- ggplot2::GeomPath$draw_panel(both, ...)
            ggplot2:::ggname(
                "geom_split_violin", grid::grobTree(
                    ggplot2::GeomPolygon$draw_panel(newdata, ...), quantile_grob))
        } else {
            ggplot2:::ggname(
                "geom_split_violin", ggplot2::GeomPolygon$draw_panel(newdata, ...))
        }
    }
)

## draw quantiles for half violins.
create_quantile_segment_frame <- function(
        data, draw_quantiles, split = FALSE, grp = NULL) {
    dens <- cumsum(data$density) / sum(data$density)
    ecdf <- stats::approxfun(dens, data$y)
    ys <- ecdf(draw_quantiles)
    violin.xminvs <- (stats::approxfun(data$y, data$xminv))(ys)
    violin.xmaxvs <- (stats::approxfun(data$y, data$xmaxv))(ys)
    violin.xs <- (stats::approxfun(data$y, data$x))(ys)
    if (grp %% 2 == 0) {
        data.frame(x = ggplot2:::interleave(violin.xs, violin.xmaxvs), 
                   y = rep(ys, each = 2), group = rep(ys, each = 2))
    } else {
        data.frame(x = ggplot2:::interleave(violin.xminvs, violin.xs), 
                   y = rep(ys, each = 2), group = rep(ys, each = 2))
    }
}

## final geom to use for plotting.
geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", 
                              position = "identity", nudge = 0, ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", 
                              na.rm = FALSE, show.legend = NA, inherit.aes = TRUE) {
    ggplot2::layer(data = data, mapping = mapping, stat = stat, 
                   geom = GeomSplitViolin, position = position, 
                   show.legend = show.legend, inherit.aes = inherit.aes, 
                   params = list(trim = trim, scale = scale, nudge = nudge, 
                                 draw_quantiles = draw_quantiles, na.rm = na.rm, 
                                 ...))
}


## plot parameters.
font.name <- "sans"
font.size <- round(c("tick" = 56, "label" = 60, "title" = 80, "plabel" = 96) / 4)
plt.clrs <- c("R" = "#E08DAC", "NR" = "#7595D0", "score1" = "#B075D0", 
              "score2" = "#C3D075", "score3" = "#FFC72C", "box" = "#A9A9A9", 
              "base" = "#000000")
dot.size <- 8 / 4
ln.size  <- 2 / 2

set.yticks <- function(y.lims) scale_y_continuous(
    breaks = seq(0, 1, by = 0.2), expand = c(0.01, 0.01), limits = y.lims)
set.clrs   <- function(clrs) scale_fill_manual(values = clrs %>% `names<-`(NULL))   # need to remove names for barplot
plt.theme  <- theme(
    panel.grid = element_blank(), 
    # panel.border = element_rect(linewidth = ln.size, color = plt.clrs["base"]), 
    axis.line = element_line(color = plt.clrs["base"], linewidth = ln.size), 
    axis.ticks = element_line(linewidth = ln.size, color = plt.clrs["base"]), 
    axis.ticks.length = unit(ln.size / 4, "cm"), 
    axis.text = element_text(size = font.size["tick"], color = plt.clrs["base"]), 
    legend.title = element_text(hjust = 0.5, face = "bold", size = font.size["label"], 
                                color = plt.clrs["base"]), 
    legend.key.size = unit(4, "line"), 
    legend.text = element_text(hjust = 0, size = font.size["tick"], 
                               color = plt.clrs["base"]))


## make figures.
fig.plotS4_I <- list()                                                                # list of all plots

## panel A + C: R vs. NR plots.
fig.plotS4_I[c("A", "C")] <- lineages %>% names %>% sapply(
    simplify = F, function(ctp){
    ggplot(data = fig.dataS4AC, mapping = aes(
        x = dataset, y = .data[[ctp]], fill = resp)) +
        geom_split_violin(stat = "ydensity", scale = "area", bw = "bcv", trim = F, 
                          na.rm = T, nudge = 0.015, linewidth = ln.size, 
                          show.legend = T) +
        geom_split_violin(stat = "ydensity", scale = "area", bw = "bcv", trim = F, 
                          na.rm = T, draw_quantiles = c(0.25, 0.5, 0.75), 
                          nudge = 0.015, linetype = "dashed", 
                          linewidth = 0.75 * ln.size, show.legend = F) +
        stat_pvalue_manual(fig.statS4AC[[ctp]], label = "p.signif", inherit.aes = F,
                           bracket.size = 0, vjust = -4,
                           label.size = font.size["tick"] / 3,
                           color = plt.clrs["base"]) +
        xlab("") + ylab(TeX(glue("{ctp}$^+$  "), bold = T)) + 
        labs(fill = "Response    ") +
        theme_classic(base_family = font.name, base_size = font.size["tick"]) +
        set.clrs(clrs = plt.clrs[c("R", "NR")]) + 
        set.yticks(y.lims = c(-0.5, 1.6)) + plt.theme + theme(
            legend.position = "left",
            axis.text.x = element_text(angle = 0, vjust = 1, hjust = 0.5,
                                       size = font.size["tick"],
                                       color = plt.clrs["base"]),
            axis.title.y = element_text(angle = 0, vjust = 0.5, face = "bold",
                                        size = font.size["title"],
                                        color = plt.clrs["base"]))
})

# print(ggarrange(fig.plotS4_I$A, fig.plotS4_I$C), nrow = 2, ncol = 1)

## panel B + D: performance plots.
fig.plotS4_I[c("B", "D")] <- lineages %>% names %>% sapply(
    simplify = F, function(ctp){
    ggplot(data = fig.dataS4BD, 
           mapping = aes(x = dataset, y = .data[[ctp]], fill = metric)) + 
        geom_bar(stat = "identity", position = "dodge", width = 0.6, 
                 linewidth = ln.size, color = plt.clrs["base"]) + 
        geom_text(mapping = aes(
            label = .data[[ctp]] %>% sapply(function(x) sprintf("%0.2f", x))), 
            position = position_dodge(width = 0.6), hjust = 0.5, vjust = -0.35, 
            size = font.size["tick"] / 3, color = plt.clrs["base"]) + 
        xlab("") + ylab("") + labs(fill = "Performance") + 
        theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
        set.clrs(clrs = plt.clrs[c("score1", "score2")]) + 
        set.yticks(y.lims = c(-0.05, 1.05)) + plt.theme + theme(
            axis.text.x = element_text(angle = 0, vjust = 1, hjust = 0.5, 
                                       size = font.size["tick"], 
                                       color = plt.clrs["base"]))
})

# print(ggarrange(fig.plotS4_I$B, fig.plotS4_I$D), nrow = 2, ncol = 1)


## combine plots.
fig.plotS4_I[["AC"]] <- ggarrange(
    fig.plotS4_I$A + theme(axis.text.x = element_blank()), fig.plotS4_I$C, 
    nrow = 2, ncol = 1, heights = c(1.0, 1.0), legend = "left", common.legend = T, 
    labels = c("A", "C"), label.x = 0.10, label.y = 1.06, font.label = list(
        family = font.name, size = font.size["plabel"], color = plt.clrs["base"])) 

fig.plotS4_I[["BD"]] <- ggarrange(
    fig.plotS4_I$B + theme(axis.text.x = element_blank()), fig.plotS4_I$D, 
    nrow = 2, ncol = 1, heights = c(1.0, 1.0), legend = "right", common.legend = T, 
    labels = c("B", "D"), label.x = -0.04, label.y = 1.06, font.label = list(
        family = font.name, size = font.size["plabel"], color = plt.clrs["base"])) 

fig.plotS4_I[["final"]] <- ggarrange(
    plotlist = fig.plotS4_I[c("AC", "BD")], nrow = 1, ncol = 2) + 
    theme(plot.margin = margin(l = -0.1, r = -0.1, t = -0.1, b = -0.1, unit = "cm"))

print(fig.plotS4_I$final)


## save plot.
svdat <- F                                                                          # set as T to save figure
if (svdat){
    fig.path     <- "../../data/TransNEO/transneo_analysis/plots/final_plots6/"
    fig.fileS4_I <- glue("all_chemo_CD4_CD8_performance_{use.ctp}.pdf")
    
    pdf(file = paste0(fig.path, fig.fileS4_I), height = 28, width = 60)
    print(fig.plotS4_I$final)
    
    dev.off()
}

