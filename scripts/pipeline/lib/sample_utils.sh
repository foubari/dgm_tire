#!/bin/bash
# sample_utils.sh - Sampling utilities for EpureDGM pipeline
#
# Functions:
# - sample_unconditional: Generate unconditional samples
# - sample_conditional: Generate conditional samples
# - sample_inpainting: Generate inpainting samples
# - sample_all_modes: Run all sampling modes for a model
# - supports_inpainting: Check if model supports inpainting

# Check if a model supports inpainting
supports_inpainting() {
    local model="$1"

    # MDM and WGAN-GP don't support inpainting
    if [[ "$model" == "mdm" ]] || [[ "$model" == "wgan_gp" ]]; then
        return 1  # Does not support
    else
        return 0  # Supports
    fi
}

# Get component list for dataset
get_components() {
    local dataset="$1"

    if [ "$dataset" = "epure" ]; then
        echo "group_nc group_km bt fpu tpc"
    else  # toy
        echo "group_nc group_km fpu"
    fi
}

# Sample unconditional
sample_unconditional() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    log_info "Sampling unconditional: $model seed$seed"

    local checkpoint=$(get_checkpoint_path "$model" "$seed" "$dataset")

    if [ ! -f "$checkpoint" ]; then
        log_error "Checkpoint not found: $checkpoint"
        return 1
    fi

    local log_file="logs/pipeline/${PIPELINE_ID}/sampling/${model}_seed${seed}_unconditional.log"

    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd python "src/models/${model}/sample.py" \
            --checkpoint "$checkpoint" \
            --mode unconditional \
            --num_samples 1000 \
            --batch_sz 64
        return 0
    fi

    # Run sampling
    python "src/models/${model}/sample.py" \
        --checkpoint "$checkpoint" \
        --mode unconditional \
        --num_samples 1000 \
        --batch_sz 64 \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Unconditional sampling completed: $model seed$seed"
        return 0
    else
        log_error "Unconditional sampling failed: $model seed$seed"
        log_error "See log: $log_file"
        return 1
    fi
}

# Sample conditional
sample_conditional() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    log_info "Sampling conditional: $model seed$seed"

    local checkpoint=$(get_checkpoint_path "$model" "$seed" "$dataset")

    if [ ! -f "$checkpoint" ]; then
        log_error "Checkpoint not found: $checkpoint"
        return 1
    fi

    local log_file="logs/pipeline/${PIPELINE_ID}/sampling/${model}_seed${seed}_conditional.log"

    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd python "src/models/${model}/sample.py" \
            --checkpoint "$checkpoint" \
            --mode conditional \
            --num_samples 1000 \
            --batch_sz 64
        return 0
    fi

    # Run sampling
    python "src/models/${model}/sample.py" \
        --checkpoint "$checkpoint" \
        --mode conditional \
        --num_samples 1000 \
        --batch_sz 64 \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Conditional sampling completed: $model seed$seed"
        return 0
    else
        log_error "Conditional sampling failed: $model seed$seed"
        log_error "See log: $log_file"
        return 1
    fi
}

# Sample inpainting (for one component)
sample_inpainting_component() {
    local model="$1"
    local seed="$2"
    local dataset="$3"
    local component="$4"

    log_info "Sampling inpainting: $model seed$seed (preserve: $component)"

    local checkpoint=$(get_checkpoint_path "$model" "$seed" "$dataset")

    if [ ! -f "$checkpoint" ]; then
        log_error "Checkpoint not found: $checkpoint"
        return 1
    fi

    local log_file="logs/pipeline/${PIPELINE_ID}/sampling/${model}_seed${seed}_inpainting_${component}.log"

    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd python "src/models/${model}/sample.py" \
            --checkpoint "$checkpoint" \
            --mode inpainting \
            --components "$component" \
            --num_samples 1000 \
            --batch_sz 64
        return 0
    fi

    # Run sampling
    python "src/models/${model}/sample.py" \
        --checkpoint "$checkpoint" \
        --mode inpainting \
        --components "$component" \
        --num_samples 1000 \
        --batch_sz 64 \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Inpainting sampling completed: $model seed$seed ($component)"
        return 0
    else
        log_error "Inpainting sampling failed: $model seed$seed ($component)"
        log_error "See log: $log_file"
        return 1
    fi
}

# Sample inpainting (all components)
sample_inpainting() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    # Check if model supports inpainting
    if ! supports_inpainting "$model"; then
        log_warn "Model $model does not support inpainting, skipping"
        return 0
    fi

    log_info "Sampling inpainting (all components): $model seed$seed"

    # Get component list
    local components=$(get_components "$dataset")

    # Sample for each component
    local failed=0
    for component in $components; do
        if ! sample_inpainting_component "$model" "$seed" "$dataset" "$component"; then
            ((failed++))
        fi
    done

    if [ $failed -eq 0 ]; then
        log_success "All inpainting sampling completed: $model seed$seed"
        return 0
    else
        log_error "$failed inpainting runs failed for $model seed$seed"
        return 1
    fi
}

# Run all sampling modes
sample_all_modes() {
    local model="$1"
    local seed="$2"
    local dataset="$3"

    log_info "Sampling all modes: $model seed$seed"

    local failed=0

    # Unconditional
    if ! sample_unconditional "$model" "$seed" "$dataset"; then
        ((failed++))
    fi

    # Conditional
    if ! sample_conditional "$model" "$seed" "$dataset"; then
        ((failed++))
    fi

    # Inpainting (if supported)
    if ! sample_inpainting "$model" "$seed" "$dataset"; then
        ((failed++))
    fi

    if [ $failed -eq 0 ]; then
        log_success "All sampling modes completed: $model seed$seed"
        return 0
    else
        log_error "$failed sampling modes failed for $model seed$seed"
        return 1
    fi
}
