#!/bin/bash
# common.sh - Shared utilities for EpureDGM pipeline scripts
#
# Functions:
# - Logging (log_info, log_success, log_warn, log_error, log_fatal, log_stage)
# - Terminal colors
# - State management
# - Time utilities

# Colors for terminal output
if [[ -t 1 ]]; then
    # Terminal supports colors
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    # No color support
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    RESET=''
fi

# Global variables
PIPELINE_ID=""
PIPELINE_START_TIME=""
LOG_FILE=""
ERROR_LOG=""
STATE_FILE=""

# Initialize pipeline environment
init_pipeline() {
    # Create pipeline ID (timestamp)
    PIPELINE_ID=$(date +"%Y-%m-%d_%H-%M-%S")
    PIPELINE_START_TIME=$(date +"%s")

    # Create log directory
    local log_dir="logs/pipeline/${PIPELINE_ID}"
    mkdir -p "$log_dir"
    mkdir -p "$log_dir/training"
    mkdir -p "$log_dir/sampling"
    mkdir -p "$log_dir/evaluation"

    # Set log files
    LOG_FILE="$log_dir/pipeline.log"
    ERROR_LOG="$log_dir/errors.log"
    STATE_FILE="$log_dir/state.json"

    # Create symlink to latest
    rm -rf "logs/pipeline/latest"
    ln -s "$PIPELINE_ID" "logs/pipeline/latest"

    # Initialize state file
    cat > "$STATE_FILE" <<EOF
{
  "pipeline_id": "$PIPELINE_ID",
  "dataset": "$DATASET",
  "start_time": "$(date +"%Y-%m-%dT%H:%M:%S")",
  "stage": "init",
  "completed": {
    "training": {},
    "sampling": {},
    "evaluation": {}
  },
  "failed": {
    "training": [],
    "sampling": [],
    "evaluation": []
  }
}
EOF

    log_info "Pipeline initialized: $PIPELINE_ID"
    log_info "Logs directory: $log_dir"
}

# Logging functions
log_info() {
    local msg="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${BLUE}[INFO]${RESET} $msg" | tee -a "$LOG_FILE"
}

log_success() {
    local msg="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${GREEN}[SUCCESS]${RESET} $msg" | tee -a "$LOG_FILE"
}

log_warn() {
    local msg="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${YELLOW}[WARN]${RESET} $msg" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${RED}[ERROR]${RESET} $msg" | tee -a "$LOG_FILE" | tee -a "$ERROR_LOG"
}

log_fatal() {
    local msg="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${RED}${BOLD}[FATAL]${RESET} $msg" | tee -a "$LOG_FILE" | tee -a "$ERROR_LOG"
    exit 1
}

log_stage() {
    local stage="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
    echo -e "${MAGENTA}${BOLD}STAGE: $stage${RESET}" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Format time duration (seconds -> human readable)
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m ${secs}s"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}m ${secs}s"
    else
        echo "${secs}s"
    fi
}

# Get elapsed time since pipeline start
get_elapsed_time() {
    local current_time=$(date +"%s")
    local elapsed=$((current_time - PIPELINE_START_TIME))
    format_duration $elapsed
}

# Progress bar (simple version)
show_progress() {
    local current=$1
    local total=$2
    local desc=$3

    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))

    printf "\r${CYAN}[$desc]${RESET} ["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %3d%% (%d/%d)" $percent $current $total
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check if GPU is available
check_gpu() {
    if command_exists nvidia-smi; then
        nvidia-smi &> /dev/null
        return $?
    else
        return 1
    fi
}

# Get available disk space in GB
get_disk_space_gb() {
    local path="${1:-.}"

    # Try Linux/Mac df
    if df -BG "$path" &> /dev/null; then
        df -BG "$path" | tail -1 | awk '{print $4}' | sed 's/G//'
    # Try Windows df
    elif df "$path" &> /dev/null; then
        local blocks=$(df "$path" | tail -1 | awk '{print $4}')
        echo $((blocks / 1024 / 1024))
    else
        echo "0"
    fi
}

# Update state file (append to completed or failed)
update_state() {
    local stage="$1"  # training, sampling, evaluation
    local model="$2"
    local seed="$3"
    local status="$4"  # completed, failed

    # This is a simplified version - in production use jq for JSON manipulation
    # For now, just log
    log_info "State: $stage/$model/seed$seed -> $status"
}

# Generate final summary
generate_final_summary() {
    local end_time=$(date +"%s")
    local duration=$((end_time - PIPELINE_START_TIME))

    echo "" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
    echo " PIPELINE SUMMARY" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
    echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "Pipeline ID: $PIPELINE_ID" | tee -a "$LOG_FILE"
    echo "Start time: $(date -d @$PIPELINE_START_TIME +"%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r $PIPELINE_START_TIME +"%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "N/A")" | tee -a "$LOG_FILE"
    echo "End time: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a "$LOG_FILE"
    echo "Total duration: $(format_duration $duration)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Count successes/failures from logs
    local training_success=$(grep -c "\[SUCCESS\].*Training OK" "$LOG_FILE" 2>/dev/null || echo 0)
    local training_failed=$(grep -c "\[ERROR\].*Training failed" "$LOG_FILE" 2>/dev/null || echo 0)

    echo "TRAINING:   $training_success succeeded, $training_failed failed" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
        echo "Errors logged to: $ERROR_LOG" | tee -a "$LOG_FILE"
    fi

    echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "=================================================================================" | tee -a "$LOG_FILE"
}

# Cleanup on exit
cleanup() {
    if [ -n "$LOG_FILE" ]; then
        log_info "Pipeline terminated"
    fi
}

trap cleanup EXIT

# Show help message
show_help() {
    cat <<EOF
EpureDGM Pipeline Script

Usage: $0 [OPTIONS]

Required:
  --dataset {epure,toy}         Dataset to use

Optional:
  --models MODEL1,MODEL2,...    Specific models (default: all 9)
  --seeds SEED1,SEED2,...       Seeds to use (default: 0,1,2)
  --skip-training               Skip training (use existing runs)
  --skip-sampling               Skip sampling (use existing samples)
  --skip-evaluation             Skip evaluation
  --resume-from MODEL           Resume from specific model
  --force                       Overwrite existing runs
  --dry-run                     Print what would be done
  -h, --help                    Show this help

Examples:
  # Full pipeline for EPURE dataset
  $0 --dataset epure

  # Only specific models
  $0 --dataset toy --models ddpm,mdm,flow_matching

  # Resume from vae onwards
  $0 --dataset epure --resume-from vae

  # Evaluate existing runs
  $0 --dataset epure --skip-training --skip-sampling
EOF
}

# Dry run execution
dry_run_cmd() {
    echo -e "${CYAN}[DRY RUN]${RESET} $*"
}

# Execute command (or dry run)
execute_cmd() {
    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd "$@"
        return 0
    else
        "$@"
        return $?
    fi
}
