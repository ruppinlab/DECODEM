# DECODEM / DECODEMi: Systematic assessment of the diverse cell types in tumor microenvironment in clinical phenotypes from bulk transcriptome  

We developed a novel computational framework called **DECODEM** (<ins>DE</ins>coupling <ins>C</ins>ell-type-specific <ins>O</ins>utcomes using <ins>DE</ins>convolution and <ins>M</ins>achine learning) that can systematically assess the roles of the diverse cell types in the tumor microenvironment (TME) in a given phenotype from bulk transcriptomics. In this work, we investigate the association of the diverse cell types in breast cancer TME (BC-TME) to patient response to neoadjuvant chemotherapy. The framework is divided into two steps:  
1. <b>Deconvolution</b>: we use [CODEFACS](https://github.com/ruppinlab/CODEFACS/) to deconvolve the bulk gene expression into nine cell-type-specific gene expression profiles encompassing malignant, immune, and stromal cell types.   
2. <b>Machine learning</b>: we use each cell-type-specific expression profile to build a machine learning predictor of chemotherapy response.   

The output of this framework is then the predictive powers of the  nine cell types (in terms of AUC and AP) which we can use to *rank* the cell types in BC-TME and validate in multiple independent cohorts encompassing both bulk and single-cell transcriptomics. Furthermore, we investigate the interactions between different cell types: (a) by incorporating the top predictive cell types that improves the predictive performance even further, or (b) by extending DECODEM to **DECODEMi** ('i' stands for interaction) where we use the inferred cell-cell interactions (CCIs) to identify the key CCIs in chemotherapy response. Our findings in breast cancer highlight the considerable predictive powers of the immune and stromal cells in TME as well as denote key CCIs that are strongly predictive of chemotherapy response.  

* The relevant manuscript is currently under review *  
Authors: Saugato Rahman Dhruba, Sahil Sahni, Binbin Wang, Di Wu, Yael Schmidt, Eldad D. Shulman, Sanju Sinha, Stephen-John Sammut, Carlos Caldas, Kun Wang, Eytan Ruppin  


![DECODEM](./figures/Fig1_DECODEM_v2.png)  
*The full analysis pipeline for DECODEM and DECODEMi*
  
<br></br>
**Contact**:  
Saugato Rahman Dhruba (saugatorahman.dhruba@nih.gov)  
Cancer Data Science Lab, CCR, NCI, NIH  
