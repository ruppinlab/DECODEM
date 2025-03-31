#### ---------------------------------------------------------------------------
#### created on 16 dec 2024, 09:56pm
#### author: dhrubas2
#### ---------------------------------------------------------------------------

setwd("/data/Lab_ruppin/projects/TME_contribution_project/analysis/analysis_final/")
source("/home/dhrubas2/vivid/miscellaneous.R")

library(latex2exp)
library(rstatix)
library(ggpubr)

fcat <- function(...) cat(paste0(glue::glue(...), "\n"))                        # f-string print akin to python

cat("\014")                                                                     # clears console


#### ---------------------------------------------------------------------------

## read data.
data.path  <- "../../data/SC_data/WuEtAl2021/"
data.files <- c("WuEtAl2021_benchmark_correlations_no_mix.RDS", 
                "WuEtAl2021_benchmark_correlations_mix.RDS", 
                "WuEtAl2021_benchmark_correlations_mix2.RDS", 
                "WuEtAl2021_benchmark_correlations_no_mix_noisy.RDS", 
                "WuEtAl2021_benchmark_correlations_no_mix_noisy2.RDS")

ds.info    <- data.frame(
    "Dataset" = paste0("BC", 1:5), 
    "n" = data.files %>% sapply(USE.NAMES = F, function(x){
        ifelse(x %>% grepl(pattern = "no_mix"), yes = 22, no = 100)
    })) %>% mutate(label = glue("{.$Dataset} (n = {.$n})"))


corr.data  <- data.files %>% sapply(simplify = F, function(file){
    readRDS(paste0(data.path, file))
}) %>% `names<-`(ds.info$label)


#### ---------------------------------------------------------------------------

## prepare data for benchmarking figures.
## cell fraction.
fig.dataS1A <- corr.data %>% sapply(simplify = F, function(dat){
    dat$corr.frac$tau %>% 
        `names<-`(dat$corr.frac %>% rownames %>% 
                      gsub(pattern = " ", replacement = "\n"))
}) %>% as.data.frame(check.names = F) %>% mutate(mean = rowMeans(.)) %>% 
    dplyr::arrange(desc(mean)) %>% select(-mean) %>% 
    rownames_to_column(var = "Cell.type") %>% 
    mutate(Cell.type = Cell.type %>% factor(levels = Cell.type)) %>% 
    reshape2::melt(id.vars = "Cell.type", variable.name = "Dataset", 
                   value.name = "tau")

cell.types <- fig.dataS1A$Cell.type %>% levels

## cell-type-specific expression.
fig.dataS1B <- ds.info$label %>% sapply(simplify = F, function(ds){
    corr.data[[ds]]$corr.exp %>% 
        sapply(function(x) x$tau %>% `names<-`(x %>% rownames)) %>% 
        `colnames<-`(colnames(.) %>% gsub(pattern = " ", replacement = "\n")) %>% 
        as.data.frame(check.names = F) %>% rownames_to_column(var = "Gene") %>% 
        reshape2::melt(id.vars = "Gene", variable.name = "Cell.type", 
                       value.name = "tau") %>% drop_na %>% 
        mutate(Dataset = ds)
}) %>% do.call(rbind, .) %>% as.data.frame(check.names = F) %>% 
    `rownames<-`(NULL) %>% 
    mutate(Cell.type = Cell.type %>% factor(levels = cell.types))

## confidently inferred genes (tau ≥ 0.3).
fig.dataS1C <- corr.data %>% sapply(simplify = F, function(dat){
    dat$corr.exp.summary$well.predicted.genes %>% 
        `names<-`(dat$corr.exp.summary %>% rownames %>% 
                      gsub(pattern = " ", replacement = "\n"))
}) %>% as.data.frame(check.names = F) %>% .[cell.types, ] %>% 
    rownames_to_column(var = "Cell.type") %>% 
    mutate(Cell.type = Cell.type %>% factor(levels = cell.types)) %>% 
    reshape2::melt(id.vars = "Cell.type", variable.name = "Dataset", 
                   value.name = "genes") %>% 
    mutate(genes = genes / 1e3)

tau.cut <- 0.3


#### ---------------------------------------------------------------------------

## plot performance.
font.name <- "sans"
font.size <- c("tick" = 12, "label" = 16, "title" = 20, "plabel" = 36)
dot.size  <- c("min" = 3, "max" = 6)
plt.clrs  <- c("#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", "#FFC72C", 
               "#708090", "bg" = "#A9A9A9", base = "#000000")
line.size <- c("axis" = 1.5, "dot" = 1, "tick" = 1)

plt.theme <- 
    theme_classic(base_family = font.name, base_size = font.size["tick"]) + 
    theme(axis.line = element_line(linewidth = line.size["axis"], 
                                   color = plt.clrs["base"]), 
          axis.ticks.length = unit(3, "mm"), 
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


## panel A: cell fraction inference.
fig.plotS1A  <- ggplot(
    data = fig.dataS1A, mapping = aes(x = Cell.type, y = tau, fill = Dataset)) + 
    geom_bar(stat = "identity", position = "dodge", width = 0.5, 
             color = plt.clrs["base"], linewidth = line.size["tick"]) + 
    # geom_hline(yintercept = tau.cut, linewidth = line.size["tick"], 
    #            linetype = "dotdash", color = plt.clrs["base"]) + 
    scale_fill_manual(values = plt.clrs[1:5] %>% `names<-`(NULL)) + 
    scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(-0.01, 1.01)) + 
    xlab("") + ylab(TeX("Kendall correlation, $\\tau$")) + 
    ggtitle("Cell fraction inference") + plt.theme

# print(fig.plotS1A)


## panel B: cell-type-specific expression inference.
fig.plotS1B <- ggplot(
    data = fig.dataS1B, mapping = aes(x = Cell.type, y = tau, fill = Dataset)) + 
    geom_violin(trim = T, linewidth = line.size["tick"], show.legend = T) + 
    geom_violin(trim = T, draw_quantiles = c(0.25, 0.5, 0.75), 
                linetype = "dashed", linewidth = line.size["tick"] * 0.5, 
                show.legend = F) + 
    geom_hline(yintercept = 0, linewidth = line.size["tick"], 
               linetype = "dotdash", color = plt.clrs["base"]) + 
    # geom_hline(yintercept = 0, linewidth = line.size["tick"], 
    #            linetype = "dotdash", color = plt.clrs["bg"]) + 
    scale_fill_manual(values = plt.clrs[1:5] %>% `names<-`(NULL)) + 
    scale_y_continuous(breaks = seq(-1.1, 1.0, by = 0.3) %>% round(1), 
                       limits = c(-0.91, 1.05)) +
    xlab("") + ylab(TeX("Kendall correlation, $\\tau$")) + 
    ggtitle("Gene expression inference") + plt.theme

# print(fig.plotS1B)


## panel C: confidently inferred genes (tau ≥ 0.3).
fig.plotS1C <- ggplot(
    data = fig.dataS1C, mapping = aes(x = Cell.type, y = genes, fill = Dataset)) + 
    geom_bar(stat = "identity", position = "dodge", width = 0.5, 
             color = plt.clrs["base"], linewidth = line.size["tick"]) + 
    # geom_hline(yintercept = 1, linewidth = line.size["tick"], 
    #            linetype = "dotdash", color = plt.clrs["base"]) + 
    scale_fill_manual(values = plt.clrs[1:5] %>% `names<-`(NULL)) + 
    scale_y_continuous(breaks = seq(0, 10, by = 2), limits = c(-0.01, 10.01)) + 
    xlab("") + ylab(TeX("Number of genes ($\\times 10^3$)")) + 
    ggtitle(TeX("Confidently inferred genes ($\\tau \\geq 0.3, \\textit{P} \\leq 0.05$)", 
                bold = T)) + plt.theme

# print(fig.plotS1C)


## combined plot.
fig.plotS1 <- ggpubr::ggarrange(
    fig.plotS1A + theme(axis.text.x = element_blank()), 
    fig.plotS1B + theme(axis.text.x = element_blank()), 
    fig.plotS1C + theme(axis.text.x = element_text(face = "bold")), 
    nrow = 3, ncol = 1, heights = c(0.9, 0.9, 1.1), align = "v", 
    labels = c("A", "B", "C"), label.x = -0.002, label.y = 1.05, hjust = 0, 
    common.legend = T, legend = "right", font.label = list(
        size = font.size["plabel"], color = plt.clrs["base"]))

print(fig.plotS1)


#### ---------------------------------------------------------------------------
 
## save plots.
svdat <- F

if (svdat){
    fig.path   <- "../../data/plots/"
    if (!(fig.path %>% dir.exists)){    dir.create(fig.path)    }
    fig.fileS1 <- glue("WuEtAl2021_benchmarking_plots.pdf")

    pdf(file = paste0(fig.path, fig.fileS1), height = 12, width = 20)
    print(fig.plotS1)

    dev.off()
    print(fig.fileS1)
}


