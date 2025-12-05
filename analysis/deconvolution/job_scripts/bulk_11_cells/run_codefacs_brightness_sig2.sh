#!/bin/bash
#SBATCH --mem=80g
#SBATCH --time=24:00:00
#SBATCH --partition=norm
#SBATCH --gres=lscratch:20
#SBATCH --output=job_codefacs_brightness_%j.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_50,TIME_LIMIT_80,FAIL


#### --------------------------------------------------------------

## get necessary variables.
## directories.
PROJ="/data/Lab_ruppin/projects/TME_contribution_project";
ODIR="$PROJ/data/BrighTNess";
RUN_CF="$PROJ/analysis/CODEFACS/CODEFACS2/scripts";
ODIR_CF="$RUN_CF/out_codefacs_brightness_v3";

## data files.
BULK="$ODIR/GSE164458_BrighTNess_RNAseq_TPM_v2_SRD_09Oct2022.csv";
SIGN="$PROJ/data/celltype_signature/signature_scSigR_BRCA_LFC3_v2.csv";

## create output directories.
if [ ! -d "$ODIR_CF" ]; then
mkdir $ODIR_CF
fi


#### --------------------------------------------------------------

module load R/4.3

## run CODEFACS with SC BRCA signature.
echo "running CODEFACS with SC derived signature... "

cd $RUN_CF
Rscript CODEFACS_v0.11.10_updated.r -t $BULK -s $SIGN -e "mem=80g,time=24:00:00" -n 15 -o $ODIR_CF

echo "done!";    echo " "


#### --------------------------------------------------------------

## copy deconvolved data to output directory.
echo "copying CODEFACS output to output directory: $ODIR/... "

cp -r $ODIR_CF $ODIR

echo "done!"

