#!/bin/bash

#SBATCH -J temp_build # job name
#SBATCH -p cpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -n 20 # number of cores
#SBATCH -t 5-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%N.%j.out # write STDOUT
#SBATCH -e slurm.%x.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ucqfnsi@ucl.ac.uk

# Load the required modules
module load miniconda
module load ants
# Activate the conda environment
conda activate template-builder

# Define atlas-forge directory, species and template names, and average type
ATLAS_DIR="/ceph/neuroinformatics/neuroinformatics/atlas-forge"
SPECIES="BlackCap"
TEMP_NAME="template_sym_res-50um_n-18"
AVE_TYPE="mean"

# Path to the bash script that builds the template
BUILD_SCRIPT="${ATLAS_DIR}/${SPECIES}/scripts/BlackCap_build_template.sh"

# Run the script to build the template
bash $BUILD_SCRIPT --atlas-dir $ATLAS_DIR --species $SPECIES --template-name $TEMP_NAME --average-type $AVE_TYPE
