#!/bin/bash

# Start the timer
start_time=$(date +%s)

# Initialize CLI parameters with default values
average_type="mean"

# Function to display help message
usage() {
  echo "Usage: $0 --atlas-dir <path> --template-name <string> [--average-type <string>]"
  echo ""
  echo "Options:"
  echo "  --atlas-dir <path>         Path to the atlas-forge directory [REQUIRED]"
  echo "  --template-name <string>   The name of the appropriate subfolder within templates (e.g., template_asym_res-50um_n-8) [REQUIRED]"
  echo "  --average-type <string>    The type of average to use (default: mean)."
  exit 1
}

# Check for help flag first
if [[ "$1" == "--help" ]]; then
  usage
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --atlas-dir)
      atlas_dir="$2"
      shift 2
      ;;
    --template-name)
      template_name="$2"
      shift 2
      ;;
    --average-type)
      average_type="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option $1"
      usage
      ;;
  esac
done

# Check for required parameters
if [ -z "$atlas_dir" ] || [ -z "$template_name" ]; then
  echo "Error: --atlas-dir and --template-name are required."
  usage
fi

echo "atlas-dir: ${atlas_dir}"
echo "Building the template ${template_name}..."

# Set the path to the working directory where the template will be built
templates_dir="${atlas_dir}/templates"
working_dir="${templates_dir}/${template_name}"

# If average type is trimmed_mean or efficient_trimean, we need python
average_prog="ANTs"
if [ "${average_type}" == "trimmed_mean" ] || [ "${average_type}" == "efficient_trimean" ]; then
  average_prog="python"
fi

# Verify that the working directory exists before changing directory
if [ ! -d "${working_dir}" ]; then
  echo "The working directory does not exist: ${working_dir}"
  exit 1
fi

cd "${working_dir}" || exit

echo "Results will be written to ${working_dir}"

# Execute the actual template building
echo "Starting to build the template..."
bash modelbuild.sh --output-dir "${working_dir}" \
    --starting-target first \
    --stages rigid,similarity,affine,nlin \
    --masks "${working_dir}/mask_paths.txt" \
    --average-type "${average_type}" \
    --average-prog "${average_prog}" \
    --reuse-affines \
    --walltime-short "03:00:00" \
    --walltime-linear "06:00:00" \
    --walltime-nonlinear "80:00:00"\
    --no-dry-run \
    "${working_dir}/brain_paths.txt"

echo "Finished building the template!"

# Write execution time (in HH:MM format) to a file
end_time=$(date +%s)
execution_time=$((end_time - start_time))
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60))
formatted_time=$(printf "%02d:%02d" $hours $minutes)
echo "Execution time: $formatted_time" > "${working_dir}/execution_time.txt"
