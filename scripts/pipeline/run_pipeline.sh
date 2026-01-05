#!/bin/bash
# run_pipeline.sh - Main pipeline script for EpureDGM
#
# Orchestrates training, sampling, and evaluation for all models

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility libraries
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/train_utils.sh"
source "${SCRIPT_DIR}/lib/sample_utils.sh"
source "${SCRIPT_DIR}/lib/eval_utils.sh"

# Default values
DATASET=""
MODELS=("ddpm" "mdm" "flow_matching" "vae" "gmrf_mvae" "meta_vae" "vqvae" "wgan_gp" "mmvaeplus")
SEEDS=(0 1 2)
SKIP_TRAINING=false
SKIP_SAMPLING=false
SKIP_EVALUATION=false
RESUME_FROM=""
FORCE=false
DRY_RUN=false

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --models)
                IFS=',' read -ra MODELS <<< "$2"
                shift 2
                ;;
            --seeds)
                IFS=',' read -ra SEEDS <<< "$2"
                shift 2
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --skip-sampling)
                SKIP_SAMPLING=true
                shift
                ;;
            --skip-evaluation)
                SKIP_EVALUATION=true
                shift
                ;;
            --resume-from)
                RESUME_FROM="$2"
                shift 2
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [ -z "$DATASET" ]; then
        echo "Error: --dataset is required"
        show_help
        exit 1
    fi

    if [[ ! "$DATASET" =~ ^(epure|toy)$ ]]; then
        echo "Error: --dataset must be 'epure' or 'toy'"
        exit 1
    fi
}

# Print pipeline configuration
print_config() {
    echo "================================================================================="
    echo " EPUREDGM PIPELINE"
    echo "================================================================================="
    echo "Dataset: $DATASET"
    echo "Models: ${MODELS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo ""
    echo "Stages:"
    echo "  - Training: $( [ "$SKIP_TRAINING" = true ] && echo "SKIP" || echo "RUN" )"
    echo "  - Sampling: $( [ "$SKIP_SAMPLING" = true ] && echo "SKIP" || echo "RUN" )"
    echo "  - Evaluation: $( [ "$SKIP_EVALUATION" = true ] && echo "SKIP" || echo "RUN" )"
    echo ""
    if [ -n "$RESUME_FROM" ]; then
        echo "Resume from: $RESUME_FROM"
        echo ""
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "MODE: DRY RUN (no actual execution)"
        echo ""
    fi
    echo "================================================================================="
    echo ""
}

# Main pipeline execution
main() {
    # Parse arguments
    parse_args "$@"

    # Print configuration
    print_config

    # Initialize pipeline (logging, state, etc.)
    init_pipeline

    # Stage 1: Training
    if [ "$SKIP_TRAINING" = false ]; then
        log_stage "TRAINING (${#MODELS[@]} models × ${#SEEDS[@]} seeds)"

        local training_count=0
        local training_failed=0
        local total_training=$((${#MODELS[@]} * ${#SEEDS[@]}))

        local resume_found=false
        if [ -z "$RESUME_FROM" ]; then
            resume_found=true
        fi

        for model in "${MODELS[@]}"; do
            # Handle resume logic
            if [ "$resume_found" = false ]; then
                if [ "$model" = "$RESUME_FROM" ]; then
                    resume_found=true
                    log_info "Resuming from model: $model"
                else
                    log_info "Skipping $model (resuming from $RESUME_FROM)"
                    continue
                fi
            fi

            for seed in "${SEEDS[@]}"; do
                ((training_count++))
                show_progress $training_count $total_training "Training"

                echo ""  # New line after progress bar

                # Check if training needed (unless force)
                if [ "$FORCE" = false ] && ! needs_training "$model" "$seed" "$DATASET"; then
                    log_info "Checkpoint exists for $model seed$seed, skipping"
                    continue
                fi

                # Train model
                if train_model_with_seed "$model" "$seed" "$DATASET"; then
                    # Create symlink
                    if ! create_run_symlink "$model" "$seed" "$DATASET"; then
                        log_warn "Failed to create symlink for $model seed$seed"
                    fi

                    # Verify checkpoint
                    if ! verify_checkpoint "$model" "$seed" "$DATASET"; then
                        log_error "Checkpoint verification failed for $model seed$seed"
                        ((training_failed++))
                    fi
                else
                    ((training_failed++))
                fi

                echo ""  # Spacing between runs
            done
        done

        echo ""  # Clear progress bar
        log_info "Training stage complete: $((training_count - training_failed))/$training_count succeeded"

        if [ $training_failed -gt 0 ]; then
            log_warn "$training_failed training runs failed"
        fi
    fi

    # Stage 2: Sampling
    if [ "$SKIP_SAMPLING" = false ]; then
        log_stage "SAMPLING (${#MODELS[@]} models × ${#SEEDS[@]} seeds × 3 modes)"

        local sampling_count=0
        local sampling_failed=0
        local total_sampling=$((${#MODELS[@]} * ${#SEEDS[@]}))

        for model in "${MODELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                ((sampling_count++))
                show_progress $sampling_count $total_sampling "Sampling"

                echo ""  # New line after progress bar

                # Check if checkpoint exists
                local checkpoint=$(get_checkpoint_path "$model" "$seed" "$DATASET")
                if [ ! -f "$checkpoint" ]; then
                    log_error "Checkpoint not found for $model seed$seed: $checkpoint"
                    ((sampling_failed++))
                    continue
                fi

                # Run all sampling modes
                if ! sample_all_modes "$model" "$seed" "$DATASET"; then
                    ((sampling_failed++))
                fi

                echo ""  # Spacing between runs
            done
        done

        echo ""  # Clear progress bar
        log_info "Sampling stage complete: $((sampling_count - sampling_failed))/$sampling_count succeeded"

        if [ $sampling_failed -gt 0 ]; then
            log_warn "$sampling_failed sampling runs failed"
        fi
    fi

    # Stage 3: Evaluation
    if [ "$SKIP_EVALUATION" = false ]; then
        log_stage "EVALUATION (${#MODELS[@]} models)"

        local eval_count=0
        local eval_failed=0
        local total_eval=${#MODELS[@]}

        for model in "${MODELS[@]}"; do
            ((eval_count++))
            show_progress $eval_count $total_eval "Evaluation"

            echo ""  # New line after progress bar

            # Evaluate all seeds
            if ! evaluate_model_all_seeds "$model" "$DATASET"; then
                ((eval_failed++))
            fi

            echo ""  # Spacing between runs
        done

        echo ""  # Clear progress bar
        log_info "Evaluation stage complete: $((eval_count - eval_failed))/$eval_count succeeded"

        if [ $eval_failed -gt 0 ]; then
            log_warn "$eval_failed evaluations failed"
        fi

        # Aggregate results
        log_info "Aggregating results..."
        aggregate_results "$DATASET"
    fi

    # Generate final summary
    generate_final_summary

    log_success "Pipeline complete! Logs saved to: logs/pipeline/${PIPELINE_ID}/"
}

# Run main
main "$@"
