# DECODEM / DECODEMi: Systematic assessment of the roles of diverse cell types in TME in clinical response from bulk transcriptome  

<i>**The relevant manuscript is currently under review:   
S. R. Dhruba, S. Sahni, B. Wang, D. Wu, Y. Schmidt, E. Shulman, S. Sinha, S. Sammut, C. Caldas, K. Wang, E. Ruppin. <b>"Predicting breast cancer patient response to neoadjuvant chemotherapy from the deconvolved tumor microenvironment transcriptome"</b>, 2023.  
</i>

We developed a novel computational framework called **DECODEM** (<ins>DE</ins>coupling <ins>C</ins>ell-type-specific <ins>O</ins>utcomes using <ins>DE</ins>convolution and <ins>M</ins>achine learning) that can systematically assess the roles of the diverse cell types in the tumor microenvironment (TME) in a given phenotype from bulk transcriptomics. In this work, we investigate the association of the diverse cell types in breast cancer TME (BC-TME) to patient response to neoadjuvant chemotherapy (R vs. NR). The framework is divided into two steps:  

1. <b>Deconvolution</b> [[see relevant codes](./analysis/deconvolution/)]: we use [CODEFACS](https://github.com/ruppinlab/CODEFACS/) to deconvolve the bulk gene expression into nine cell-type-specific gene expression profiles encompassing malignant, immune, and stromal cell types.  
2. <b>Machine Learning</b> [[see relevant codes](./analysis/machine_learning/)]: we use a four-stage **ML pipeline** to build nine cell-type-specific predictors of chemotherapy response using the deconvolved expression profiles.    

The output of the framework is the cell-type-specific predictive powers (in terms of AUC and AP) which we use to *rank* the cell types in BC-TME and externally validate in multiple independent cohorts encompassing both bulk and single-cell transcriptomics.  

![DECODEM](./figures/Fig1_DECODEM_v2.png)  
*Figure: The full analysis pipeline for DECODEM and DECODEMi*  
  
Furthermore, we investigate the interactions between different cell types in two ways:  
* <b><i>Multi-cell-ensemble</i></b>: we incorporate the expression profiles of the top predictive cell types to boost the predictive power even further, yielding the best performance for an ensemble of immune and stromal cell types across two independent cohorts.  
* <b><i>DECODEMi</i></b>: we extended DECODEM to **DECODEMi** ('i' stands for interaction) where we use the inferred cell-cell interactions (CCIs) to identify the cellular communications that influence chemotherapy response.  

Our findings in breast cancer highlight the considerable predictive powers of the immune and stromal cells in the TME as well as denote key CCIs that are strongly predictive of chemotherapy response.  


## Dependencies  
For `python` scripts:  
```
python >= 3.8  
numpy >= 1.23   
pandas >= 1.4  
scikit-learn >= 1.0.2  
xgboost 1.6.1
pickle >= 3.0  
matplotlib >= 3.7
seaborn >= 0.12
tqdm >= 4.63  
lifelines >= 0.27  
```  
  
For `R` scripts:  
```
R >= 3.6  
tidyverse >= 1.3  
plyr >= 1.8
rtracklayer >= 1.57  
GenomicFeatures >= 1.50
clusterProfiler >= 4.6  
biomaRt >= 2.54  
msigdbr >= 7.5  
GSVA >= 1.45  
PRROC >= 1.3  
rstatix >= 0.7  
ggpubr >= 0.6  
glue >= 1.6  
```


## Reproducing the results
All the main results presented in the above manuscript can be reproduced by using the codes in [analysis/machine_learning](analysis/machine_learning/). This assumes that the bulk expression data has already been deconvolved and put inside [data](data/), which can be achieved by using the scripts in [analysis/deconvolution](analysis/deconvolution/).  

### DECODEM  
- To perform the cross-validation analysis using the TransNEO cohort, use the script: `model_transneo_cv_v1.py`  
- To train the cell-type-specific / multi-cell-ensemble predictors using TransNEO and validate on the ARTemis + PBCP cohort, use the script: `predict_sammut_validation_v2.py`  
- To train the cell-type-specific / multi-cell-ensemble predictors using TransNEO and validate on the BrighTNess cohort, use the script: `predict_brightness_validation_v2.py`  
- To train the cell-type-specific predictors using TransNEO and validate on the Zhang et al. single-cell cohort, use the script: `predict_tnbc_sc_validation_v2.py`  
- To train the cell-type-specific predictors using TransNEO and stratify survival on the TCGA-BRCA cohort, use the script: `stratify_tcga_validation_v3.py` 


#### DECODEMi  
- To perform the cross-validation analysis using the TransNEO cohort and extract the top predictive CCIs, use the script: `model_transneo_lirics_cv_v3.py`  
- To train the CCI-based predictors using TransNEO and validate on the ARTemis + PBCP cohort, use the script: `predict_sammut_lirics_validation_v2.py`  
- To train the CCI-based predictors using TransNEO and validate on the BrighTNess cohort, use the script: `predict_brightness_lirics_validation_v2.py`  
- To computationally validate the top CCIs for prediction in TNBC that were extracted from DECODEMi using the single-cell pseudopatient cohort generated from the Zhang et al. SC-TNBC cohort (generates Figs. S4E-F),  use the script: `predict_sc_validation_cci_pseudopatients_v1.R`  


### Enrichment & other analyses  
The enrichment analyses results and all the figures/panels in the manuscript can be reproduced using the codes in [analysis/enrichment_and_figures](analysis/enrichment_and_figure/).  
- To perform the cell-type-specific GSEA analysis (generates Fig. 3E), use the script: `run_enrichment_top_cell_types_v3.R`
- To perform the GSVA analysis for CD4+ / CD8+ T-cells and estimate their predictive power (generates Fig. S3A-D), use the script: `enrichment_cd4_cd8_tcells_v2.R`   
- To perform the association analysis between cell-type-abundance and chemotherapy response (generates Fig. S3E-G), use the script: `get_abundance_response_corr_v2.py`  


### Reproducing the figures  
To reproduce the figures, use the following scripts in [analysis/enrichment_and_figures](analysis/enrichment_and_figures/):  
- Figs. 2, 3A-D, S1-2: `generate_plots_ctp_v2.py`  
- Figs. 4, S4A-D: `generate_plots_cci_v2.py`  
- Figs. 5, S5: `generate_plots_sc_surv_v2.py`  

Examples of the generated figures are provided in [figures](figures/). 


#### Data preprocessing  
All the datasets should be deposited in [data](data/) using the structure outlined. To preprocess data into the desired formats, use the scripts in [analysis/preprocessing](analysis/preprocessing/).  
Examples of the processed datasets are provided in: [data/TransNEO](data/TransNEO/) and [data/BrighTNess](data/BrighTNess/). 


  
<br></br>
### Contact
Saugato Rahman Dhruba (saugatorahman.dhruba@nih.gov)  
Cancer Data Science Lab, National Cancer Institute, National Institutes of Health  
