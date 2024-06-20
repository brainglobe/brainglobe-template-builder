#!/bin/bash

# This script is used to build the a BlackCap brain template on a
# desktop computer (not HPC). It is assumed that the user has already
# run the BlackCap_source_images.py and BlackCap_build_template.py scripts.

# Start the timer
start_time=$(date +%s)

# Initialize CLI parameters with default values
atlas_dir="/ceph/neuroinformatics/neuroinformatics/atlas-forge"
species=""
template_name=""
average_type="median"

# Function to display help message
usage() {
  echo "Usage: $0 --atlas_dir <path> --species <string> --template_name <string> --average-type <string>"
  echo ""
  echo "Options:"
  echo "  --atlas_dir <path>         Path to the atlas-forge directory (default: /ceph/neuroinformatics/neuroinformatics/atlas-forge)"
  echo "  --species <string>         Species name (e.g., BlackCap)"
  echo "  --template_name <string>   The name of the appropriate subfolder within templates (e.g., template_asym_res-50um_n-8)"
  echo "  --average-type <string>    The type of average to use (default: median)."
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --atlas_dir)
      atlas_dir="$2"
      shift
      shift
      ;;
    --species)
      species="$2"
      shift
      shift
      ;;
    --template_name)
      template_name="$2"
      shift
      shift
      ;;
    ---average_type)
      average_type="$2"
      shift
      shift
      ;;
    --help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

# Check if all required parameters are set
if [ -z "$atlas_dir" ] || [ -z "$species" ] || [ -z "$template_name" ]; then
  usage
fi

echo "atlas_dir: ${atlas_dir}"
echo "Building the template ${template_name} for the species ${species}..."

# Set the path to the working directory where the template will be built
species_dir="${atlas_dir}/${species}"
templates_dir="${species_dir}/templates"
working_dir="${templates_dir}/${template_name}"
cd "${working_dir}" || exit

echo "Results will be written to ${working_dir}"

# verify that the working directory exists
if [ ! -d "${working_dir}" ]; then
    echo "The working directory does not exist: ${working_dir}"
    exit 1
fi

# Execute the actual template building

echo "Starting to build the template..."
bash modelbuild.sh --output-dir "${working_dir}" \
    --starting-target first \
    --stages rigid,similarity,affine,nlin \
    --masks "${working_dir}"/mask_paths.txt \
    --average-type "${average_type}" \
    --reuse-affines \
    --dry-run \
    "${working_dir}"/brain_paths.txt

echo "Finished building the template!"

# Write execution time (in HH:MM format) to a file
end_time=$(date +%s)
execution_time=$((end_time - start_time))
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60))
formatted_time=$(printf "%02d:%02d" $hours $minutes)
echo "Execution time: $formatted_time"
