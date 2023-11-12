#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=80g
#SBATCH --gres=lscratch:20
#SBATCH --partition=ccr
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_50,TIME_LIMIT_80,FAIL


#### --------------------------------------------------------------

## get necessary variables.
## directories.
PROJ="/data/Lab_ruppin/projects/TME_contribution_project";
ODIR="$PROJ/data/TransNEO";
RUN_CF="$PROJ/analysis/CODEFACS/CODEFACS2/scripts";
ODIR_CF="$RUN_CF/CODEFACS_results";

## data files.
BULK="$ODIR/transneo-diagnosis-RNAseq-TPM_SRD_26May2022.tsv";
SIGN="$PROJ/data/celltype_signature/signature_scSigR_BRCA.csv";


## create output directories.
if [ ! -d "$ODIR_CF" ]; then
mkdir $ODIR_CF
fi


#### --------------------------------------------------------------                                    

module load R/3.6

## run CODEFACS with sc BRCA signature.
echo "running CODEFACS with sc BRCA signature... "

cd $RUN_CF
Rscript CODEFACS_v0.11.10.r -t $BULK -s $SIGN -e "mem=80g,time=48:00:00,partition=ccr" -n 15 -o $ODIR_CF

echo "done!";    echo " "


## copy deconvolved data to output directory.
echo "copying CODEFACS output to output directory: $ODIR... "

cp -r $ODIR_CF $ODIR

echo "done!"
