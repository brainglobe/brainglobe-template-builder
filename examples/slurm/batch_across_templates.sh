#!/bin/bash

#SBATCH -J temp_build # job name
#SBATCH -p cpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 128G # memory pool for all cores
#SBATCH -n 20 # number of cores
#SBATCH -t 5-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%N.%j.out # write STDOUT
#SBATCH -e slurm.%x.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.sirmpilatze@ucl.ac.uk
#SBATCH --array=0-2

# Load the required modules
module load miniconda
module load ants
# Activate the conda environment
conda activate template-builder

# Define the atlas-forge directory and species name
ATLAS_DIR="/ceph/neuroinformatics/neuroinformatics/atlas-forge"
SPECIES="BlackCap"

# Define the template names as an array
TEMP_NAME1="template_asym_res-50um_n-8"
TEMP_NAME2="template_sym_res-50um_n-18"
TEMP_NAME3="template_hemi_res-50um_n-18"
TEMP_NAMES=($TEMP_NAME1 $TEMP_NAME2 $TEMP_NAME3)

CURRENT_TEMP_NAME=${TEMP_NAMES[$SLURM_ARRAY_TASK_ID]}

# Path to the bash script that builds the template
BUILD_SCRIPT="${ATLAS_DIR}/${SPECIES}/scripts/BlackCap_build_template.sh"

# Run the script to build the template
bash $BUILD_SCRIPT --atlas-dir $ATLAS_DIR --species $SPECIES --template-name $CURRENT_TEMP_NAME
