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
# Activate the conda environment (becaue we need parallel and qbatch)
conda activate template-builder

# Define the atlas-forge directory and species name
ATLAS_DIR="/ceph/neuroinformatics/neuroinformatics/atlas-forge"
SPECIES="BlackCap"

# Define the average types as an array
AVE_TYPES=(median trimmed_mean efficient_trimean)
AVE_TYPE=${AVE_TYPES[$SLURM_ARRAY_TASK_ID]}

# Original template created via mean averaging
TEMP_NAME_ORIG="template_asym_res-50um_n-8"

# get only the fitst 4 characters of the average type
AVE_TYPE_SHORT=$(echo $AVE_TYPE | cut -c1-4)
# Define the template name as
TEMP_NAME="${TEMP_NAME_ORIG}_average-${AVE_TYPE_SHORT}"
# create the template folder
mkdir -p ${ATLAS_DIR}/${SPECIES}/templates/${TEMP_NAME}
# Copy the brain_paths.txt and mask_paths.txt files from the original template
cp ${ATLAS_DIR}/${SPECIES}/templates/${TEMP_NAME_ORIG}/brain_paths.txt ${ATLAS_DIR}/${SPECIES}/templates/${TEMP_NAME}
cp ${ATLAS_DIR}/${SPECIES}/templates/${TEMP_NAME_ORIG}/mask_paths.txt ${ATLAS_DIR}/${SPECIES}/templates/${TEMP_NAME}

# Path to the bash script that builds the template
BUILD_SCRIPT="${ATLAS_DIR}/${SPECIES}/scripts/BlackCap_build_template.sh"

# Run the script to build the template
bash $BUILD_SCRIPT --atlas_dir $ATLAS_DIR --species $SPECIES --template_name $TEMP_NAME --average-type $AVE_TYPE
