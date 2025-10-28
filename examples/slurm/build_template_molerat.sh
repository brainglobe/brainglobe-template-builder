#!/bin/bash

#SBATCH -J slurm_molerat_parallel # job name
#SBATCH -p cpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -n 10 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%N.%j.out # write STDOUT
#SBATCH -e slurm.%x.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.felder@ucl.ac.uk

# Load the required modules
module load template-builder/2024-12-02

export QBATCH_PPJ=12
export QBATCH_CHUNKSIZE=1
export QBATCH_CORES=1
export QBATCH_SYSTEM="slurm"
export QBATCH_QUEUE="cpu"
export QBATCH_MEM="128G"
export QBATCH_OPTIONS="--mail-type=ALL --mail-user=a.felder@ucl.ac.uk --mem 128G"

# Define atlas-forge directory, species and template names, and average type
ATLAS_DIR="/ceph/neuroinformatics/neuroinformatics/atlas-forge"
SPECIES="MoleRat"
TEMP_NAME="template_sym_res-20um_n-43_avg-trimean"
AVE_TYPE="efficient_trimean"

# Path to the bash script that builds the template
BUILD_SCRIPT="${ATLAS_DIR}/${SPECIES}/scripts/4_build_template.sh"

# Run the script to build the template
bash $BUILD_SCRIPT --atlas-dir "${ATLAS_DIR}/${SPECIES}" --template-name $TEMP_NAME --average-type $AVE_TYPE
