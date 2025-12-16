#!/bin/bash

###
# Wrapper script that is intended to be called from atlas-specific configuration slurm files
# and calls modelbuild.sh under the hood.
# Does some basic checks and prints basic logs to standard out.
###

# Start the timer
start_time=$(date +%s)

echo  "Starting build_slurm.sh."

# Initialize CLI parameters with default values
average_type="mean"
walltime_short="1:30:00"
walltime_linear="03:00:00"
walltime_nonlinear="40:00:00"
toggle_dry_run="--dry-run"


# Function to display help message
usage() {
  echo "Usage: $0 --template-dir <path> [--average-type <string> --walltime-short <string> --walltime-linear <string> --walltime-nonlinear <string> --(no-)dry-run]"
  echo ""
  echo "Options:"
  echo "  --template-dir <path>         Path to the atlas-forge template directory [REQUIRED]"
  echo "  --average-type <string>       The type of average to use (default: mean)."
  echo "  --walltime-short <string>     The max time required for "short" averaging/shape_update steps in HH:MM:SS (default: 1:30:00)."
  echo "  --walltime-linear <string>    The max time required for linear registration steps in HH:MM:SS (default: 3:00:00)."
  echo "  --walltime-nonlinear <string> The max time required for nonlinear registration steps in HH:MM:SS (default: 40:00:00)."
  echo "  --(no-)dry-run                Toggle whether a dry run should be executed or not (default: --dry-run)."
  exit 1
}

# Check for help flag first
if [[ "$1" == "--help" ]]; then
  usage
fi

echo "Parsing command line arguments."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --template-dir)
      template_dir="$2"
      shift 2
      ;;
    --average-type)
      average_type="$2"
      shift 2
      ;;
    --walltime-short)
      walltime_short="$2"
      shift 2
      ;;
    --walltime-linear)
      walltime_linear="$2"
      shift 2
      ;;
    --walltime-nonlinear)
      walltime_nonlinear="$2"
      shift 2
      ;;
    --toggle-dry-run)
      toggle_dry_run="$2"
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

echo "Finished parsing."

# Check for required parameters
if [ -z "$template_dir" ]; then
  echo "Error: --template-dir is required."
  usage
fi

echo "template-dir: ${template_dir}"
echo "Building the template ${template_name}..."

# If average type is trimmed_mean or efficient_trimean, we need python
average_prog="ANTs"
if [ "${average_type}" == "trimmed_mean" ] || [ "${average_type}" == "efficient_trimean" ]; then
  average_prog="python"
fi

# Verify that the working directory exists before changing directory
if [ ! -d "${template_dir}" ]; then
  echo "${template_dir} does not exist"
  exit 1
fi

cd $template_dir || exit

echo "Results will be written to ${template_dir}"

# Execute the actual template building
echo "Starting to build the template..."
bash modelbuild.sh --output-dir "${template_dir}" \
    --starting-target first \
    --stages rigid,similarity,affine,nlin \
    --average-type "${average_type}" \
    --average-prog "${average_prog}" \
    --reuse-affines \
    --walltime-short "${walltime_short}" \
    --walltime-linear "${walltime_linear}" \
    --walltime-nonlinear "${walltime_nonlinear}" \
    "${toggle_dry_run}" \
    --masks "${template_dir}/all_processed_mask_paths.txt" \
    "${template_dir}/all_processed_brain_paths.txt" 

echo "Finished building the template!"

# Write execution time (in HH:MM format) to a file
end_time=$(date +%s)
execution_time=$((end_time - start_time))
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60))
formatted_time=$(printf "%02d:%02d" $hours $minutes)
echo "Execution time: $formatted_time" > "${template_dir}/execution_time.txt"

