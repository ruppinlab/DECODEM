#### -------------------------------------------------------------------------------
#### created on 19 may 2023, 08:42pm
#### author: dhrubas2
#### -------------------------------------------------------------------------------

.wpath. <- "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
.mpath. <- "miscellaneous/r/miscellaneous.R"
setwd(.wpath.)                                                                      # current path
source(.mpath.)

library(rstatix)
library(clusterProfiler)
library(org.Hs.eg.db)
library(biomaRt)
library(msigdbr)

fcat <- function(...) cat(paste0(glue(...), "\n"))

## enrichment functions.g
enrichREACTOME <- ReactomePA::enrichPathway
gseREACTOME    <- ReactomePA::gsePathway
viewREACTOME   <- ReactomePA::viewPathway

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
data.path <- "../data/TransNEO/transneo_analysis/mdl_data/"
data.file <- c("data_tn_tn_valid_bn_conf_genes_exp_resp_SRD.RDS", 
               "transneo_RvsNR_wilcox_pvals_SRD_22May2023.RDS")

data.list <- readRDS(paste0(data.path, data.file[1]))

## transneo.
exp.all.tn      <- data.list$transneo_exp
exp.all.conf.tn <- data.list$transneo_exp_conf
conf.all.tn     <- data.list$transneo_conf
resp.pCR.tn     <- data.list$transneo_resp

## transneo valid.
exp.all.tn.val      <- data.list$transneo_valid_exp
exp.all.conf.tn.val <- data.list$transneo_valid_exp_conf
conf.all.tn.val     <- data.list$transneo_valid_conf
resp.pCR.tn.val     <- data.list$transneo_valid_resp

## brightness.
exp.all.bn      <- data.list$brightness_exp
exp.all.conf.bn <- data.list$brightness_exp_conf
conf.all.bn     <- data.list$brightness_conf
resp.pCR.bn     <- data.list$brightness_resp

data.list %>% rm                                                                    # release memory


## parameters.
cell.types     <- exp.all.tn %>% names
cell.types.int <- c("Cancer_Epithelial", "Myeloid", "Plasmablasts", "B-cells",
                    "Endothelial", "Normal_Epithelial")
pval.cut       <- 0.2
eps            <- 1e-4                                                              # nonzero min. value cut-off
num.perm       <- 1e3                                                               # #permutations used for gsea


#### -------------------------------------------------------------------------------

## gsea for transneo.
lddat <- T                                                                          # load pre-computed data 
if (lddat){
    data.list <- readRDS(paste0(data.path, data.file[2]))
    pvals.all.conf.tn <- data.list$conf
    pvals.all.tn      <- data.list$all.raw
    p.adj.all.tn      <- data.list$all
    
    data.list %>% rm
} else {
    pvals.all.conf.tn <- get.pval.resp(exp.all.conf.tn, resp.pCR.tn, alt.hyp = "g") # R vs. NR wilcoxon test

    pvals.all.tn <- get.pval.resp(exp.all.tn, resp.pCR.tn, alt.hyp = "g")           # R vs. NR wilcoxon test
    p.adj.all.tn <- data.frame(pvals.all.tn, check.names = F) / (conf.all.tn + eps) # adjust p-values by confidence levels
}

gsea.all.tn  <- get.gsea.cell.types(p.adj.all.tn, 
                                    cell.types = cell.types.int, 
                                    use.pathway = "reactome", 
                                    pval.cut = pval.cut, 
                                    nPermSimple = num.perm, 
                                    eps = eps)

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


#### -------------------------------------------------------------------------------

## save data.
svdat <- F                                                                          # set T to save data 
if (svdat){
    datestamp <- DateTime()                                                         # datestamp for analysis
    
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

## prepare data for fig 3E.
num.path.disp <- 15                                                                 # #pathways per cell type to plot
fig.data3 <- cell.types.int %>% sapply(simplify = F, function(ctp){
    gsea.all.tn[[ctp]] %>% mutate(
        Pathway = Description %>% gsub(pattern = "  ", replacement = " ", fixed = T), 
        logP = -log(p.adjust), 
        cell.type = ctp %>% gsub(pattern = "_", replacement = "\n")) %>% 
        dplyr::arrange(desc(logP), pvalue, desc(NES %>% abs)) %>% 
        (function(df) df[!grepl(df$ID, pattern = "R-HSA-9"), ]) %>%                 # remove non-relevant pathways (SARS-COV)
        head(num.path.disp) %>% dplyr::select(Pathway, NES, logP, cell.type)
}) %>% Reduce(f = rbind) %>% data.frame(check.names = F) 


#### -------------------------------------------------------------------------------

## make gsea plot - fig 3E.
font.name <- "sans"
font.size <- round(c("tick" = 56, "label" = 60, "title" = 84) / 4)                  # set denominator to 1 when saving the plot
dot.size  <- round(c("min" = 16, "max" = 28) / 2)                                   # set denominator to 1 when saving the plot
plt.clrs  <- c("min" = "#EFCC74", "max" = "#DC91AD", "base" = "#000000")            # ("jasmine", "amaranth pink", "black")
line.size <- c("axis" = 4, "dot" = 3, "legend" = 2) / 4                             # set denominator to 1 when saving the plot
cbar.size <- round(c("h" = 2, "w" = 1.8) / 2.5, 1)                                  # set denominator to 1 when saving the plot
plt.theme <- theme(
    panel.grid = element_blank(), 
    panel.border = element_rect(linewidth = line.size["axis"], 
                                color = plt.clrs["base"]), 
    axis.ticks = element_line(linewidth = line.size["axis"] / 2, 
                              color = plt.clrs["base"]), 
    axis.text = element_text(size = font.size["label"], color = plt.clrs["base"]), 
    legend.title = element_text(size = font.size["label"], face = "bold", 
                                color = plt.clrs["base"]), 
    legend.title.align = 0.45, 
    legend.text = element_text(size = font.size["label"], color = plt.clrs["base"]), 
    legend.text.align = 0)


fig.plot3 <- ggplot(
    data = fig.data3, mapping = aes(x = cell.type, y = Pathway, size = logP)) + 
    geom_point(mapping = aes(color = NES)) + geom_point(
        shape = 21, stroke = line.size["dot"], color = plt.clrs["base"]) + 
    xlab("") + ylab("") + scale_size_continuous(
        name = latex2exp::TeX("$-\\bf{log}P$", bold = T, italic = T), 
        range = dot.size) + 
    scale_color_gradient(
        low = plt.clrs["min"], high = plt.clrs["max"], 
        limits = fig.data3$NES %>% range %>% round, guide = guide_colorbar(
            frame.colour = plt.clrs["base"], ticks.colour = plt.clrs["base"], 
            frame.linewidth = line.size["legend"], 
            ticks.linewidth = line.size["legend"])) + 
    guides(size = guide_legend(order = 1), fill = guide_legend(order = 2)) + 
    theme_bw(base_family = font.name, base_size = font.size["tick"]) + plt.theme + 
    theme(axis.text.x = element_text(size = font.size["label"], angle = 40, 
                                     color = plt.clrs["base"], hjust = 1, vjust = 1), 
          legend.key.height = unit(cbar.size["h"], "cm"), 
          legend.key.width = unit(cbar.size["w"], "cm"))

print(fig.plot3)


## save plot.
svdat <- F                                                                          # set T to save figure 
if (svdat){
    datestamp <- DateTime()                                                         # datestamp for analysis
    fig.path  <- "../data/plots/"
    fig.file3 <- glue("transneo_chemo_gsea_reactome_pathways_all_v3_{datestamp}.pdf")
    
    pdf(file = paste0(fig.path, fig.file3), height = 52, width = 60)
    print(fig.plot3)
    
    dev.off()
}

