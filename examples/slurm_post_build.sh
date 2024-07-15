#!/bin/bash

#SBATCH -J postbuild # job name
#SBATCH -p cpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -n 16 # number of cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%N.%j.out # write STDOUT
#SBATCH -e slurm.%x.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niko.sirbiladze@gmail.com

# Load the required modules
module load miniconda
module load ants
# Activate the conda environment
conda activate atlas-forge

# Run the post-build Python script
python BlackCap_post_build.py
