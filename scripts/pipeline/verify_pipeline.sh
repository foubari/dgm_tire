#!/bin/bash
# verify_pipeline.sh - Pre-flight verification for EpureDGM pipeline
#
# Checks:
# - Configs exist
# - Data directories exist
# - GPU available
# - Disk space sufficient
# - Python dependencies

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "${SCRIPT_DIR}/lib/common.sh"

# Parse arguments
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --dataset {epure|toy}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    exit 1
fi

echo "================================================================================="
echo " EPUREDGM PIPELINE VERIFICATION"
echo "================================================================================="
echo "Dataset: $DATASET"
echo "================================================================================="
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Check 1: Configs exist
echo "Checking configuration files..."
MODELS=("ddpm" "mdm" "flow_matching" "vae" "gmrf_mvae" "meta_vae" "vqvae" "wgan_gp" "mmvaeplus")
CONFIGS_MISSING=0

for model in "${MODELS[@]}"; do
    config="src/configs/pipeline/${DATASET}/${model}_${DATASET}_pipeline.yaml"
    if [ ! -f "$config" ]; then
        echo "  ✗ Missing: $config"
        ((CONFIGS_MISSING++))
    fi
done

if [ $CONFIGS_MISSING -eq 0 ]; then
    echo "  ✓ All configuration files found"
    ((CHECKS_PASSED++))
else
    echo "  ✗ $CONFIGS_MISSING configuration files missing"
    echo "    Run: python scripts/create_pipeline_configs.py"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 2: Data directories
echo "Checking data directories..."
if [ "$DATASET" = "epure" ]; then
    DATA_ROOT="data/epure"
else
    DATA_ROOT="data/toy_epure"
fi

DATA_OK=true

if [ ! -d "${DATA_ROOT}/train" ]; then
    echo "  ✗ Training data not found: ${DATA_ROOT}/train"
    DATA_OK=false
fi

if [ ! -d "${DATA_ROOT}/test" ]; then
    echo "  ✗ Test data not found: ${DATA_ROOT}/test"
    DATA_OK=false
fi

if [ ! -f "${DATA_ROOT}/../performances.csv" ] && [ ! -f "${DATA_ROOT}/performances.csv" ]; then
    echo "  ✗ Performances CSV not found"
    DATA_OK=false
fi

if [ "$DATA_OK" = true ]; then
    echo "  ✓ Data directories found"
    ((CHECKS_PASSED++))
else
    echo "  ✗ Some data files missing"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 3: GPU
echo "Checking GPU..."
if command_exists nvidia-smi; then
    if nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "  ✓ Found $GPU_COUNT GPU(s)"

        # Check VRAM
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "    VRAM: ${VRAM} MB"

        if [ "$VRAM" -lt 8000 ]; then
            echo "    ⚠ Low VRAM (<8GB), consider reducing batch sizes"
        fi

        ((CHECKS_PASSED++))
    else
        echo "  ✗ nvidia-smi failed"
        ((CHECKS_FAILED++))
    fi
else
    echo "  ✗ nvidia-smi not found (no GPU detected)"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 4: Disk space
echo "Checking disk space..."
AVAILABLE_GB=$(get_disk_space_gb ".")

if [ "$AVAILABLE_GB" -gt 150 ]; then
    echo "  ✓ Sufficient disk space: ${AVAILABLE_GB}GB available"
    ((CHECKS_PASSED++))
elif [ "$AVAILABLE_GB" -gt 80 ]; then
    echo "  ⚠ Limited disk space: ${AVAILABLE_GB}GB available (recommended: 150GB)"
    ((CHECKS_PASSED++))
else
    echo "  ✗ Insufficient disk space: ${AVAILABLE_GB}GB available (need: 150GB)"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 5: Python dependencies
echo "Checking Python dependencies..."
PYTHON_OK=true

# Check Python is available
if ! command_exists python; then
    echo "  ✗ Python not found"
    PYTHON_OK=false
else
    PYTHON_VERSION=$(python --version 2>&1)
    echo "    Python: $PYTHON_VERSION"
fi

# Check key packages
REQUIRED_PACKAGES=("torch" "yaml" "PIL" "numpy" "pandas")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $pkg" &> /dev/null; then
        echo "  ✗ Missing Python package: $pkg"
        PYTHON_OK=false
    fi
done

if [ "$PYTHON_OK" = true ]; then
    echo "  ✓ Python dependencies available"
    ((CHECKS_PASSED++))
else
    echo "  ✗ Some Python dependencies missing"
    ((CHECKS_FAILED++))
fi
echo ""

# Estimate time
echo "Estimating runtime..."
MODELS_COUNT=${#MODELS[@]}
SEEDS_COUNT=3
TOTAL_RUNS=$((MODELS_COUNT * SEEDS_COUNT))

# Rough estimates (hours per run)
AVG_TRAIN_HOURS=3
AVG_SAMPLE_HOURS=1
AVG_EVAL_HOURS=0.2

TOTAL_HOURS=$(echo "$TOTAL_RUNS * ($AVG_TRAIN_HOURS + $AVG_SAMPLE_HOURS + $AVG_EVAL_HOURS)" | bc)

echo "  Training runs: $TOTAL_RUNS"
echo "  Estimated total time: ~${TOTAL_HOURS} hours (~$(echo "$TOTAL_HOURS / 24" | bc) days)"
echo ""

# Summary
echo "================================================================================="
echo " VERIFICATION SUMMARY"
echo "================================================================================="
echo "Passed: $CHECKS_PASSED"
echo "Failed: $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✓ All checks passed! Ready to run pipeline."
    echo ""
    echo "Next steps:"
    echo "  1. Run test pipeline (quick validation):"
    echo "     ./scripts/pipeline/run_pipeline_test.sh --dataset $DATASET --models ddpm"
    echo ""
    echo "  2. Run full pipeline:"
    echo "     ./scripts/pipeline/run_pipeline.sh --dataset $DATASET"
    echo ""
    exit 0
else
    echo "✗ $CHECKS_FAILED check(s) failed. Please fix issues before running pipeline."
    echo ""
    exit 1
fi
