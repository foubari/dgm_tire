#!/bin/bash
# train_utils.sh - Training utilities for EpureDGM pipeline
#
# Functions:
# - train_model_with_seed: Train a model with specific seed
# - create_run_symlink: Create stable symlink for run
# - verify_checkpoint: Verify checkpoint exists

# Train a model with a specific seed
train_model_with_seed() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    log_info "Training $model (seed=$seed, dataset=$dataset)"

    # Determine config path
    local config="src/configs/pipeline/${dataset}/${model}_${dataset}_pipeline.yaml"

    if [ ! -f "$config" ]; then
        log_error "Config not found: $config"
        return 1
    fi

    # Determine log file
    local log_file="logs/pipeline/${PIPELINE_ID}/training/${model}_seed${seed}.log"

    # Record start time
    local start_time=$(date +"%s")

    # Train the model
    log_info "  Config: $config"
    log_info "  Log: $log_file"

    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd python "src/models/${model}/train.py" \
            --config "$config" \
            --seed "$seed"
        return 0
    fi

    # Run training
    python "src/models/${model}/train.py" \
        --config "$config" \
        --seed "$seed" \
        > "$log_file" 2>&1

    local exit_code=$?
    local end_time=$(date +"%s")
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        log_success "Training completed: $model seed$seed ($(format_duration $duration))"
        update_state "training" "$model" "$seed" "completed"
        return 0
    else
        log_error "Training failed: $model seed$seed (exit code: $exit_code)"
        log_error "See log: $log_file"
        update_state "training" "$model" "$seed" "failed"
        return 1
    fi
}

# Create symlink for a trained run
create_run_symlink() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    log_info "Creating symlink for $model seed$seed"

    # Determine output base directory
    local suffix=""
    if [ "$dataset" = "toy" ]; then
        suffix="_toy"
    fi
    local output_base="outputs/${model}${suffix}"

    if [ ! -d "$output_base" ]; then
        log_error "Output directory not found: $output_base"
        return 1
    fi

    # Find the most recent run directory (timestamped)
    local latest_dir=$(ls -dt "${output_base}"/20* 2>/dev/null | head -1)

    if [ -z "$latest_dir" ]; then
        log_error "No timestamped run directory found in $output_base"
        return 1
    fi

    # Create symlink
    local symlink="${output_base}/run_seed${seed}"

    # On Windows, we might need different symlink handling
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows: Use relative path for compatibility
        local target=$(basename "$latest_dir")
        if [ -L "$symlink" ] || [ -e "$symlink" ]; then
            rm -f "$symlink"
        fi
        # Try symbolic link, fallback to hardlink if permission denied
        ln -s "$target" "$symlink" 2>/dev/null || \
            cp -r "$latest_dir" "$symlink"
    else
        # Unix: Standard symlink
        local target=$(basename "$latest_dir")
        if [ -L "$symlink" ]; then
            rm "$symlink"
        fi
        ln -s "$target" "$symlink"
    fi

    if [ $? -eq 0 ]; then
        log_success "Created symlink: $symlink -> $(basename $latest_dir)"
        return 0
    else
        log_error "Failed to create symlink: $symlink"
        return 1
    fi
}

# Verify checkpoint exists
verify_checkpoint() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    # Determine output base directory
    local suffix=""
    if [ "$dataset" = "toy" ]; then
        suffix="_toy"
    fi
    local output_base="outputs/${model}${suffix}"

    # Check symlink
    local symlink="${output_base}/run_seed${seed}"
    if [ ! -e "$symlink" ]; then
        log_error "Symlink not found: $symlink"
        return 1
    fi

    # Check checkpoint file
    local checkpoint="${symlink}/check/checkpoint_best.pt"
    if [ ! -f "$checkpoint" ]; then
        log_error "Checkpoint not found: $checkpoint"
        return 1
    fi

    log_success "Checkpoint verified: $checkpoint"
    return 0
}

# Get checkpoint path for a model/seed
get_checkpoint_path() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    local suffix=""
    if [ "$dataset" = "toy" ]; then
        suffix="_toy"
    fi

    echo "outputs/${model}${suffix}/run_seed${seed}/check/checkpoint_best.pt"
}

# Check if training is needed (checkpoint doesn't exist)
needs_training() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    local checkpoint=$(get_checkpoint_path "$model" "$seed" "$dataset")

    if [ -f "$checkpoint" ]; then
        return 1  # Checkpoint exists, no training needed
    else
        return 0  # Checkpoint missing, training needed
    fi
}
