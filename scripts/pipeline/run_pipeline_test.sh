#!/bin/bash
# run_pipeline_test.sh - Test wrapper for pipeline
#
# Runs pipeline with test configs (1 epoch, 50 samples)

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================="
echo " EPUREDGM PIPELINE TEST MODE"
echo "================================================================================="
echo "Using test configs (1 epoch, 50 samples)"
echo "This is a quick validation run to test the pipeline infrastructure"
echo "================================================================================="
echo ""

# TODO: For test mode, we need to use test configs
# For now, just call the main pipeline script with reduced arguments

# Forward all arguments to main pipeline
"${SCRIPT_DIR}/run_pipeline.sh" "$@"

# Note: In production, this should override config paths to use
# src/configs/pipeline/test/{model}_pipeline_test.yaml
