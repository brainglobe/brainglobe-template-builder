#!/bin/bash

#SBATCH -J ave_type # job name
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

# Activate the conda environment (because we need parallel and qbatch)
conda activate template-builder

# Define the atlas-forge directory and species name
ATLAS_DIR="/ceph/neuroinformatics/neuroinformatics/atlas-forge"
SPECIES="BlackCap"

# Define the average types as an array
AVE_TYPES=(median trimmed_mean efficient_trimean)
AVE_TYPE=${AVE_TYPES[$SLURM_ARRAY_TASK_ID]}

# Original template created via mean averaging
TEMP_NAME_PREFIX="template_asym_res-50um_n-8"
# Get only the first 4 characters of the average type
AVE_TYPE_SHORT=$(echo $AVE_TYPE | cut -c1-4)
# Define the new template name
TEMP_NAME="${TEMP_NAME_PREFIX}_average-${AVE_TYPE_SHORT}"

# Path to the bash script that builds the template
BUILD_SCRIPT="${ATLAS_DIR}/${SPECIES}/scripts/BlackCap_build_template.sh"

# Run the script to build the template
bash "$BUILD_SCRIPT" --atlas-dir "$ATLAS_DIR" --species "$SPECIES" --template-name "$TEMP_NAME" --average-type "$AVE_TYPE"
