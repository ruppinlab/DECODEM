setwd("/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/")
source("/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/r/miscellaneous.R")

library(rstatix)
library(clusterProfiler)
library(org.Hs.eg.db)
library(biomaRt)
library(msigdbr)

fcat <- function(...) cat(paste0(glue(...), "\n"))                                  # f-string print akin to python

## enrichment functions.
enrichREACTOME <- ReactomePA::enrichPathway
gseREACTOME    <- ReactomePA::gsePathway
viewREACTOME   <- ReactomePA::viewPathway


## perform gsea.
get.gsea <- function(gene.list, use.pathway, go.onto = "BP", pval.cut = 0.05, 
                     pval.adjust = "fdr", eps = 1e-5, min.gs.size = 5, 
                     max.gs.size = 500, nPermSimple = 1e4, seed = 86420, 
                     method = "fgsea"){
    gsea.res <- if (use.pathway %>% toupper == "GO"){
        gseGO(geneList = gene.list, 
              OrgDb = org.Hs.eg.db, keyType = "ENTREZID", 
              ont = go.onto, eps = eps, nPermSimple = nPermSimple, 
              pvalueCutoff = pval.cut, pAdjustMethod = pval.adjust, 
              minGSSize = min.gs.size, maxGSSize = max.gs.size, 
              seed = seed, by = method)
    } else if (use.pathway %>% toupper == "KEGG"){
        gseKEGG(geneList = gene.list, 
                organism = "hsa", keyType = "kegg", 
                eps = eps, nPermSimple = nPermSimple, 
                pvalueCutoff = pval.cut, pAdjustMethod = pval.adjust, 
                minGSSize = min.gs.size, maxGSSize = max.gs.size, 
                seed = seed, by = method)
    } else if (use.pathway %>% toupper == "REACTOME"){
        gseREACTOME(geneList = gene.list, 
                    organism = "human", eps = eps, nPermSimple = nPermSimple, 
                    pvalueCutoff = pval.cut, pAdjustMethod = pval.adjust, 
                    minGSSize = min.gs.size, maxGSSize = max.gs.size, 
                    seed = seed, by = method)
    }
    gsea.res
}


## perform over-representation analysis.
get.go.enrichment <- function(gene.list, background.list, go.onto = "BP", 
                              pval.cut = 0.05, qval.cut = 0.1, pval.adjust = "fdr", 
                              min.gs.size = 5, max.gs.size = 500){
    enrichGO(gene = gene.list, universe = background.list, 
             OrgDb = org.Hs.eg.db, keyType = "ENTREZID", 
             ont = go.onto, # pool = (go.onto == "ALL"), 
             pvalueCutoff = pval.cut, qvalueCutoff = qval.cut, 
             pAdjustMethod = pval.adjust, 
             minGSSize = min.gs.size, maxGSSize = max.gs.size)
}


## provides R vs. NR p-values for a set of cell types.
get.pval.resp <- function(exp.all, resp, alt.hyp = "greater", p.adjust = "fdr"){
    ## format response data.
    resp <- resp %>% factor(levels = c(1, 0)) %>% `levels<-`(c("R", "NR"))
    
    ## compute p-values.
    pvals.all <- exp.all %>% sapply(simplify = F, function(exp.ctp){
        pb <- ProgressBar(N = exp.ctp %>% nrow)
        exp.ctp %>% apply(MARGIN = 1, function(exp.ctp.gn){
            pb$tick()
            if (exp.ctp.gn %>% var > 0){
                data.test <- data.frame(exp = exp.ctp.gn, resp = resp)
                res.test  <- wilcox_test(
                    exp ~ resp, data = data.test, comparison = c("R", "NR"), 
                    alternative = alt.hyp, p.adjust.method = p.adjust)
                res.test$p
            } else {
                NaN
            }
        })
    })
    pvals.all
}


## performs gsea for a list of cell types.
get.gsea.cell.types <- function(pvals.all, cell.types = "all", 
                                use.pathway = "reactome", go.onto = "BP", 
                                pval.cut = 0.05, pval.adjust = "fdr", eps = 1e-5, 
                                min.gs.size = 5, max.gs.size = 500, 
                                nPermSimple = 1e4, seed = 86420, method = "fgsea"){
    if (cell.types[1] == "all"){
        cell.types <- pvals.all %>% names
    }
    
    gsea.all <- cell.types %>% sapply(simplify = F, function(ctp){
        gene.ranks <- pvals.all[ctp] %>% as.data.frame %>% rownames_to_column %>% 
            `colnames<-`(c("gene", "pval")) %>% mutate(
                ID = mapIds(keys = gene, keytype = "SYMBOL", x = org.Hs.eg.db, 
                            column = "ENTREZID"), 
                logP = -log(pval)) %>% 
            filter(((ID %>% is.na) | (logP %>% is.na)) %>% `!`) %>% 
            dplyr::arrange(desc(logP))
        
        gene.list <- gene.ranks$logP %>% `names<-`(gene.ranks$ID) 
        gsea.ctp  <- get.gsea(gene.list = gene.list, use.pathway = use.pathway, 
                              go.onto = go.onto, pval.cut = pval.cut, 
                              pval.adjust = pval.adjust, min.gs.size = min.gs.size, 
                              max.gs.size = max.gs.size, nPermSimple = nPermSimple, 
                              seed = seed, method = method)
        gsea.ctp@result %>% as.data.frame
    })
    
    gsea.all
}


## summarizes gsea results for a list of cell types.
summarize.gsea.cell.types <- function(gsea.all, sort.by = "logP"){
    gsea.summary <- gsea.all %>% names %>% sapply(simplify = F, function(ctp){
        gsea.all[[ctp]] %>% mutate(Pathway = Description) %>% 
            dplyr::select(Pathway, NES, p.adjust) %>% rownames_to_column("ID") %>% 
            mutate(logP = -log(p.adjust), cell.type = ctp)
    }) %>% Reduce(f = rbind) %>% as.data.frame %>% 
        mutate(Freq = Pathway %>% sapply(function(pw) sum(Pathway == pw))) %>% 
        dplyr::arrange(desc(sort.by), desc(NES %>% abs))
    
    gsea.summary
}


fcat("\014")                                                                        # clears console


#### -------------------------------------------------------------------------------

## read data.
data.path <- "../../data/TransNEO/transneo_analysis/mdl_data/"
data.file <- "data_tn_tn_valid_bn_conf_genes_exp_resp_SRD.RDS"

data.list <- readRDS(paste0(data.path, data.file))

## transneo data.
exp.all.tn      <- data.list$transneo_exp
exp.all.conf.tn <- data.list$transneo_exp_conf
conf.all.tn     <- data.list$transneo_conf
resp.pCR.tn     <- data.list$transneo_resp

data.list %>% rm                                                                    # release memory


## parameters.
cell.types      <- exp.all.tn %>% names
cell.types.int  <- c("Cancer_Epithelial", "Endothelial", "Plasmablasts", 
                     "Normal_Epithelial", "Myeloid", "B-cells", "CAFs")
pval.cut        <- 0.2
eps             <- 1e-4                                                             # nonzero min. value cut-off
num.perm        <- 1e3                                                              # #permutations used for gsea


#### -------------------------------------------------------------------------------

## gsea for transneo.
lddat <- T
if (lddat){
    pval.file <- "transneo_RvsNR_wilcox_pvals_SRD_22May2023.RDS"
    data.list <- readRDS(paste0(data.path, pval.file))
    
    pvals.all.conf.tn <- data.list$conf
    pvals.all.tn      <- data.list$all.raw
    p.adj.all.tn      <- data.list$all
    
    data.list %>% rm                                                                # release memory
} else {
    pvals.all.tn <- get.pval.resp(exp.all.tn, resp.pCR.tn, alt.hyp = "g")           # R vs. NR wilcoxon test
    p.adj.all.tn <- data.frame(pvals.all.tn, check.names = F) / (conf.all.tn + eps) # adjust p-values by confidence levels
}

gsea.all.tn  <- get.gsea.cell.types(p.adj.all.tn, 
                                    cell.types  = cell.types.int, 
                                    use.pathway = "reactome", 
                                    pval.cut    = pval.cut, 
                                    nPermSimple = num.perm, 
                                    eps         = eps)

gsea.summary.tn <- summarize.gsea.cell.types(gsea.all.tn, sort.by = "Freq")
fcat("top pathways for TransNEO = ");       print(gsea.summary.tn %>% Head)


## make pathway x cell-type freq matrix for convenience.
pb <- ProgressBar(N = gsea.summary.tn$Pathway %>% unique %>% length)
gsea.sum.mat.tn <- gsea.summary.tn$Pathway %>% unique %>% sapply(function(pw){
    pb$tick()
    cell.types %>% sapply(function(ctp){
        gsea.summary.tn %>% filter(cell.type == ctp) %>% 
            (function(df) pw %in% df$Pathway) %>% as.numeric
    })
}) %>% t %>% data.frame(check.names = F) %>% (function(df){
    pw.ord <- gsea.summary.tn %>% dplyr::select(Pathway, Freq) %>% 
        unique %>% dplyr::arrange(desc(Freq))
    df[pw.ord$Pathway, ]
}) %>% mutate(Freq = rowSums(.))

fcat("pathway x cell-type matrix = ");      print(gsea.sum.mat.tn %>% Head(5, 10))


## save data.
svdat <- F
if (svdat){
    datestamp <- DateTime()
    
    ## p-values.
    out.file <- glue("transneo_RvsNR_wilcox_pvals_SRD_{datestamp}.RDS")
    out.list <- list("conf" = pvals.all.conf.tn, "all" = p.adj.all.tn, 
                     "all.raw" = pvals.all.tn)
    saveRDS(out.list, file = paste0(data.path, out.file), compress = T)
    
    ## gsea pathways.
    out.file <- glue("transneo_RvsNR_wilcox_pathways_SRD_{datestamp}.RDS")
    out.list <- list("pathways" = gsea.all.tn, "summary" = gsea.summary.tn, 
                     "sum.mat" = gsea.sum.mat.tn)
    saveRDS(out.list, file = paste0(data.path, out.file), compress = T)
}


#### -------------------------------------------------------------------------------

## make gsea plot.
## plot parameters.
font.name <- "sans"
font.size <- c("tick" = 20, "label" = 24, "title" = 32, "plabel" = 60) / 1.0
dot.size  <- c("min" = 4, "max" = 10) / 1.0
plt.clrs  <- c("min" = "#7595D0", "max" = "#E08DAC", "base" = "#000000")            # ("vista blue", "amaranth pink", "black")
line.size <- c("axis" = 2, "dot" = 1, "tick" = 1)
cbar.size <- c("h" = 1, "w" = 1)
plt.theme <- theme(
    panel.grid = element_blank(),
    # panel.grid = element_line(
    #     color = "#A9A9A9",linewidth = line.size["tick"] / 2, linetype = "dashed"), 
    panel.border = element_rect(
        linewidth = line.size["axis"], color = plt.clrs["base"]), 
    axis.ticks = element_line(
        linewidth = line.size["tick"], color = plt.clrs["base"]), 
    axis.ticks.length = unit(line.size["tick"] / 2, "cm"), 
    axis.text = element_text(size = font.size["label"], color = plt.clrs["base"]), 
    plot.title = element_text(
        hjust = 0.5, size = font.size["title"], face = "bold", 
        color = plt.clrs["base"]), 
    legend.title = element_text(
        hjust = 0.5, size = font.size["title"], face = "bold", 
        color = plt.clrs["base"]), 
    legend.text = element_text(hjust = 0.5, size = font.size["label"], 
                               color = plt.clrs["base"]))

fig.ttl3_II <- "Enriched Reactome pathways"


## prepare plot data.
cell.types.disp <- c("Cancer_Epithelial", "Myeloid", "Normal_Epithelial", 
                     "Plasmablasts", "Endothelial", "B-cells")
num.path.disp   <- 15                                                               # #pathways per cell type to plot

fig.data3_II <- cell.types.disp %>% sapply(simplify = F, function(ctp){
    gsea.all.tn[[ctp]] %>% mutate(
        Pathway = Description %>% gsub(pattern = "  ", replacement = " ", fixed = T), 
        logP = -log(p.adjust), 
        cell.type = ctp %>% gsub(pattern = "_", replacement = " ")) %>% 
        dplyr::arrange(desc(logP), pvalue, desc(NES %>% abs)) %>% 
        (function(df) df[!grepl(df$ID, pattern = "R-HSA-9"), ]) %>%                 # remove non-relevant pathways (SARS-COV)
        head(num.path.disp) %>% dplyr::select(Pathway, NES, logP, cell.type)
}) %>% Reduce(f = rbind) %>% data.frame(check.names = F) %>% mutate(
    cell.type = cell.type %>% factor(levels = cell.types.disp %>% 
                                         gsub(pattern = "_", replacement = " ")))


## pathway vs. cell type dotplot.
fig.plot3_II <- ggplot(
    data = fig.data3_II, mapping = aes(x = cell.type, y = Pathway, size = logP)) + 
    geom_point(mapping = aes(color = NES)) + geom_point(
        shape = 21, stroke = line.size["dot"], color = plt.clrs["base"]) + 
    xlab("") + ylab("") + ggtitle(fig.ttl3_II) + scale_size_continuous(
        name = latex2exp::TeX("$-\\bf{log}P$", bold = T, italic = T), 
        range = dot.size) + 
    scale_color_gradient(
        low = plt.clrs["min"], high = plt.clrs["max"], 
        limits = fig.data3_II$NES %>% range %>% round, guide = guide_colorbar(
            frame.colour = plt.clrs["base"], ticks.colour = plt.clrs["base"], 
            frame.linewidth = line.size["tick"], 
            ticks.linewidth = line.size["tick"])) + 
    guides(size = guide_legend(order = 1), fill = guide_legend(order = 2)) + 
    theme_bw(base_family = font.name, base_size = font.size["tick"]) + 
    plt.theme + theme(
        axis.text.x = element_text(size = font.size["label"], angle = 40, hjust = 1, 
                                   vjust = 1, color = plt.clrs["base"]), 
        legend.key.height = unit(cbar.size["h"], "cm"), 
        legend.key.width = unit(cbar.size["w"], "cm")) + 
    scale_x_discrete(labels = scales::label_wrap(10))

print(fig.plot3_II)


## save plot.
svdat <- F                                                                          # set as T to save figure
if (svdat){
    fig.path     <- "../../data/TransNEO/transneo_analysis/plots/final_plots6/"
    fig.file3_II <- glue("transneo_chemo_gsea_reactome_pathways_top.pdf")
    
    # pdf(file = paste0(fig.path, fig.file3_II), height = 20, width = 24)
    ggsave(path = fig.path, filename = fig.file3_II, plot = fig.plot3_II, 
           device = "pdf", dpi = 600, height = 20, width = 24, units = "in")
    print(fig.plot3_II)
    
    dev.off()
}

