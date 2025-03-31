#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=80g
#SBATCH --gres=lscratch:20
#SBATCH --partition=ccr
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_50,TIME_LIMIT_80,FAIL
#SBATCH --output=codefacs_bm_no_mix_noisy2_%j.out

​
#### --------------------------------------------------------------

## get necessary variables.
## directories.
PROJ="/data/Lab_ruppin/projects/TME_contribution_project";
ODIR="$PROJ/data/SC_data/WuEtAl2021";
RUN_CF="$PROJ/analysis/CODEFACS/CODEFACS2/scripts";
ODIR_CF="$RUN_CF/out_codefacs_bm_no_mix_noisy2";

## data files.
BULK="$ODIR/WuEtAl2021_benchmark_bulk_tpm_no_mix_noisy2.tsv";
SIGN="$ODIR/WuEtAl2021_benchmark_signature.txt";


## create output directories.
if [ ! -d "$ODIR_CF" ]; then
mkdir $ODIR_CF
fi


#### --------------------------------------------------------------

module load R/4.3

## run CODEFACS with SC BRCA signature.
echo "running CODEFACS with SC derived signature... "

cd $RUN_CF
Rscript CODEFACS_v0.11.10_updated.r -t $BULK -s $SIGN -e "mem=80g,time=96:00:00" -n 15 -o $ODIR_CF

echo "done!";    echo " "


#### --------------------------------------------------------------

## copy deconvolved data to output directory.
echo "copying CODEFACS output to output directory: $ODIR/... "

cp -r $ODIR_CF $ODIR

echo "done!"

