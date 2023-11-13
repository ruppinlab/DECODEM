# DECODEM / DECODEMi: Systematic assessment of the diverse cell types in tumor microenvironment in clinical phenotypes from bulk transcriptome  

We developed a novel computational framework called **DECODEM** (<ins>DE</ins>coupling <ins>C</ins>ell-type-specific <ins>O</ins>utcomes using <ins>DE</ins>convolution and <ins>M</ins>achine learning) that can systematically assess the roles of the diverse cell types in the tumor microenvironment (TME) in a given phenotype from bulk transcriptomics. In this work, we investigate the association of the diverse cell types in breast cancer TME (BC-TME) to patient response to neoadjuvant chemotherapy (R vs. NR). The framework is divided into two steps:  

1. <b>Deconvolution</b> [see [relevant codes](./analysis/deconvolution/)]: we use [CODEFACS](https://github.com/ruppinlab/CODEFACS/) to deconvolve the bulk gene expression into nine cell-type-specific gene expression profiles encompassing malignant, immune, and stromal cell types.  
2. <b>Machine Learning</b> [see [relevant codes](./analysis/machine_learning/)]: we use a four-stage **ML pipeline** to build nine cell-type-specific predictors of chemotherapy response using the deconvolved expression profiles.    

The output of the framework is the cell-type-specific predictive powers (in terms of AUC and AP) which we use to *rank* the cell types in BC-TME and externally validate in multiple independent cohorts encompassing both bulk and single-cell transcriptomics.  

![DECODEM](./figures/Fig1_DECODEM_v2.png)  
*Figure: The full analysis pipeline for DECODEM and DECODEMi*  
  
Furthermore, we investigate the interactions between different cell types in two ways:  
* <b><i>Multi-cell-ensemble</i></b>: we incorporate the expression profiles of the top predictive cell types to boost the predictive power even further, yielding the best performance for an ensemble of immune and stromal cell types across two independent cohorts.  
* <b><i>DECODEMi</i></b>: we extended DECODEM to **DECODEMi** ('i' stands for interaction) where we use the inferred cell-cell interactions (CCIs) to identify the cellular communications that influence chemotherapy response.  

Our findings in breast cancer highlight the considerable predictive powers of the immune and stromal cells in the TME as well as denote key CCIs that are strongly predictive of chemotherapy response.  

***The relevant manuscript is currently under review*  
Authors: Saugato Rahman Dhruba, Sahil Sahni, Binbin Wang, Di Wu, Yael Schmidt, Eldad D. Shulman, Sanju Sinha, Stephen-John Sammut, Carlos Caldas, Kun Wang, Eytan Ruppin  


  
<br></br><ul>
<b>Contact</b>: Saugato Rahman Dhruba (saugatorahman.dhruba@nih.gov)  
Cancer Data Science Lab, National Cancer Institute, National Institutes of Health  
</ul>