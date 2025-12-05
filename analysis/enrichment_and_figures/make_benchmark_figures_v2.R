#### ---------------------------------------------------------------------------
#### created on 16 dec 2024, 09:56pm
#### author: dhrubas2
#### ---------------------------------------------------------------------------

if (Sys.info()["sysname"] == "Darwin"){                                             # mac
    .wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/"
    .mapth. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/r/miscellaneous.R"
} else if (Sys.info()["sysname"] == "Linux"){                                       # biowulf
    .wpath. <- "/data/Lab_ruppin/projects/TME_contribution_project/analysis/LIRICS_and_SOCIAL/CellChat/"
    .mpath. <- "/home/dhrubas2/vivid/miscellaneous.R"
}

setwd(.wpath.)
source(.mapth.)

library(rstatix)
library(ggpubr)
library(latex2exp)


## functions.
fcat <- function(..., end = "\n") cat(paste0(glue(...), end))                       # f-string print akin to python

cat("\014")                                                                         # clears console


#### ---------------------------------------------------------------------------

## read data.
data.path  <- "../../data/SC_data/WuEtAl2021/results/"
data.files <- c("WuEtAl2021_benchmark_correlations_no_mix.RDS", 
                "WuEtAl2021_benchmark_correlations_mix.RDS", 
                "WuEtAl2021_benchmark_correlations_mix2.RDS", 
                "WuEtAl2021_benchmark_correlations_no_mix_noisy.RDS", 
                "WuEtAl2021_benchmark_correlations_no_mix_noisy2.RDS")

ds.info    <- data.frame(
    "Dataset" = paste0("BC", 1:5), 
    "n"       = data.files %>% sapply(USE.NAMES = F, function(x){
        ifelse(x %>% grepl(pattern = "no_mix"), yes = 22, no = 100) 
    })) %>% mutate(label = glue("{.$Dataset} (n = {.$n})"))


corr.data  <- data.files %>% sapply(simplify = F, function(file){
    readRDS(paste0(data.path, file))
}) %>% `names<-`(ds.info$label)


#### ---------------------------------------------------------------------------

## prepare data for supp fig 1. 
## panel A: cell fraction.
fig.dataS1A <- corr.data %>% sapply(simplify = F, function(dat){
    dat$corr.frac$tau %>% `names<-`(dat$corr.frac %>% rownames %>% 
                                        gsub(pattern = " ", replacement = "\n"))
}) %>% as.data.frame(check.names = F) %>% 
    mutate(mean = rowMeans(.)) %>% dplyr::arrange(desc(mean)) %>% select(-mean) %>% 
    rownames_to_column(var = "Cell.type") %>% 
    mutate(Cell.type = Cell.type %>% factor(levels = Cell.type)) %>% 
    reshape2::melt(id.vars = "Cell.type", variable.name = "Dataset", 
                   value.name = "tau")

cell.types <- fig.dataS1A$Cell.type %>% levels


## panel B: cell-type-specific expression.
fig.dataS1B <- ds.info$label %>% sapply(simplify = F, function(ds){
    corr.data[[ds]]$corr.exp %>% 
        sapply(function(x) x$tau %>% `names<-`(x %>% rownames)) %>% 
        `colnames<-`(colnames(.) %>% gsub(pattern = " ", replacement = "\n")) %>% 
        as.data.frame(check.names = F) %>% 
        rownames_to_column(var = "Gene") %>% 
        reshape2::melt(id.vars = "Gene", variable.name = "Cell.type", 
                       value.name = "tau") %>% 
        drop_na %>% 
        mutate(Dataset = ds)
}) %>% do.call(rbind, .) %>% 
    as.data.frame(check.names = F) %>% `rownames<-`(NULL) %>% 
    mutate(Cell.type = Cell.type %>% factor(levels = cell.types))


## panel C: confidently inferred genes (tau ≥ 0.3).
fig.dataS1C <- corr.data %>% sapply(simplify = F, function(dat){
    dat$corr.exp.summary$well.predicted.genes %>% 
        `names<-`(dat$corr.exp.summary %>% rownames %>% 
                      gsub(pattern = " ", replacement = "\n"))
}) %>% as.data.frame(check.names = F) %>% 
    .[cell.types, ] %>% rownames_to_column(var = "Cell.type") %>% 
    mutate(Cell.type = Cell.type %>% factor(levels = cell.types)) %>% 
    reshape2::melt(id.vars = "Cell.type", variable.name = "Dataset", 
                   value.name = "genes") %>% 
    mutate(genes = genes / 1e3)                                                     # scale count by thousands

# tau.cut <- 0.3


#### ---------------------------------------------------------------------------

## make supp fig 1: benchmarking plots.
## plot parameters.
font.name <- "sans"
font.size <- c("tick" = 12, "label" = 16, "title" = 20, "plabel" = 36)
dot.size  <- c("min" = 3, "max" = 6)
plt.clrs  <- c("#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", "#FFC72C", 
               "#708090", "bg" = "#A9A9A9", base = "#000000")
line.size <- c("axis" = 1.5, "dot" = 1, "tick" = 1)

plt.theme <- theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    theme(axis.line = element_line(linewidth = line.size["axis"], 
                                   color = plt.clrs["base"]), 
          axis.ticks.length = unit(line.size["tick"] / 4, "cm"), 
          axis.ticks = element_line(linewidth = line.size["axis"], 
                                    color = plt.clrs["base"]), 
          axis.title = element_text(size = font.size["label"]), 
          axis.text = element_text(size = font.size["label"]), 
          axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1), 
          axis.title.x = element_text(size = font.size["label"]), 
          axis.title.y = element_text(size = font.size["label"]), 
          legend.title = element_text(hjust = 0.5, size = font.size["label"], 
                                      face = "bold", color = plt.clrs["base"]), 
          legend.key.size = unit(1.5, "line"), 
          legend.text = element_text(hjust = 0, size = font.size["tick"], 
                                     color = plt.clrs["base"]), 
          plot.title = element_text(hjust = 0.5, size = font.size["title"], 
                                    face = "bold", color = plt.clrs["base"]))


## make figures.
fig.plotS1 <- list()                                                                # list of all plots

## panel A: cell fraction barplot.
fig.plotS1[["A"]] <- ggplot(
    data = fig.dataS1A, mapping = aes(x = Cell.type, y = tau, fill = Dataset)) + 
    geom_bar(stat = "identity", position = "dodge", width = 0.5, 
             color = plt.clrs["base"], linewidth = line.size["tick"]) + 
    scale_fill_manual(values = plt.clrs[1:5] %>% `names<-`(NULL)) + 
    scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(-0.01, 1.01)) + 
    xlab("") + ylab(TeX("Kendall correlation, $\\tau$")) + 
    ggtitle("Cell fraction inference") + plt.theme

# print(fig.plotS1$A)


## panel B: gene expression violin plot.
fig.plotS1[["B"]] <- ggplot(
    data = fig.dataS1B, mapping = aes(x = Cell.type, y = tau, fill = Dataset)) + 
    geom_violin(trim = T, linewidth = line.size["tick"], show.legend = T) + 
    geom_violin(trim = T, quantiles = c(0.25, 0.5, 0.75), linetype = "solid", 
                linewidth = line.size["tick"] / 2, quantile.linetype = "dashed", 
                quantile.linewidth = line.size["tick"] / 3, show.legend = F) + 
    geom_hline(yintercept = 0, linewidth = line.size["tick"], 
               linetype = "dotdash", color = plt.clrs["base"]) + 
    scale_fill_manual(values = plt.clrs[1:5] %>% `names<-`(NULL)) + 
    scale_y_continuous(breaks = seq(-1.1, 1.0, by = 0.3) %>% round(1), 
                       limits = c(-0.91, 1.05)) +
    xlab("") + ylab(TeX("Kendall correlation, $\\tau$")) + 
    ggtitle("Gene expression inference") + plt.theme

# print(fig.plotS1$B)


## panel C: confident genes barplot.
fig.plotS1[["C"]] <- ggplot(
    data = fig.dataS1C, mapping = aes(x = Cell.type, y = genes, fill = Dataset)) + 
    geom_bar(stat = "identity", position = "dodge", width = 0.5, 
             color = plt.clrs["base"], linewidth = line.size["tick"]) + 
    scale_fill_manual(values = plt.clrs[1:5] %>% `names<-`(NULL)) + 
    scale_y_continuous(breaks = seq(0, 10, by = 2), limits = c(-0.01, 10.01)) + 
    xlab("") + ylab(TeX("Number of genes ($\\times 10^3$)")) + 
    ggtitle(TeX("Confidently inferred genes ($\\tau \\geq 0.3, \\textit{P} \\leq 0.05$)", 
                bold = T)) + plt.theme

# print(fig.plotS1$C)


## final plot.
fig.plotS1[["final"]] <- ggarrange(
    fig.plotS1$A + theme(axis.text.x = element_blank()), 
    fig.plotS1$B + theme(axis.text.x = element_blank()), 
    fig.plotS1$C + theme(axis.text.x = element_text(face = "bold")), 
    nrow = 3, ncol = 1, heights = c(0.9, 0.9, 1.1), align = "v", 
    labels = c("A", "B", "C"), label.x = -0.002, label.y = 1.05, hjust = 0, 
    common.legend = T, legend = "right", font.label = list(
        size = font.size["plabel"], color = plt.clrs["base"]))

print(fig.plotS1$final)


#### -------------------------------------------------------------------------------

## save plot.
svdat <- F                                                                          # set as T to save figure

if (svdat){
    fig.path   <- "../../data/TransNEO/transneo_analysis/plots/final_plots7/"
    fig.fileS1 <- "WuEtAl2021_benchmarking_plots.pdf"
    
    ggsave(path = fig.path, filename = fig.fileS1, plot = fig.plotS1$final, 
           device = "pdf", dpi = 600, height = 12, width = 20, units = "in")
    print(fig.plotS1)
    
    fcat(fig.fileS1)
    dev.off()
}


