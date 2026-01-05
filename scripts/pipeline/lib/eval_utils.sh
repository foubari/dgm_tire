#!/bin/bash
# eval_utils.sh - Evaluation utilities for EpureDGM pipeline
#
# Functions:
# - evaluate_model_all_seeds: Evaluate all seeds for a model
# - aggregate_results: Aggregate evaluation results

# Evaluate all seeds for a model
evaluate_model_all_seeds() {
    local model="$1"
    local dataset="$2"

    log_info "Evaluating $model (all seeds) on $dataset"

    local log_file="logs/pipeline/${PIPELINE_ID}/evaluation/${model}.log"

    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd python src/scripts/evaluate.py \
            --model "$model" \
            --dataset "$dataset" \
            --seeds 0,1,2
        return 0
    fi

    # Run evaluation
    python src/scripts/evaluate.py \
        --model "$model" \
        --dataset "$dataset" \
        --seeds 0,1,2 \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Evaluation completed: $model"
        return 0
    else
        log_error "Evaluation failed: $model"
        log_error "See log: $log_file"
        return 1
    fi
}

# Aggregate all evaluation results
aggregate_results() {
    local dataset="$1"

    log_info "Aggregating evaluation results for $dataset"

    if [ "$DRY_RUN" = true ]; then
        dry_run_cmd python src/scripts/aggregate_results.py \
            --dataset "$dataset" \
            --output "evaluation_results/${dataset}/summary.json"
        return 0
    fi

    # Run aggregation
    python src/scripts/aggregate_results.py \
        --dataset "$dataset" \
        --output "evaluation_results/${dataset}/summary.json"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "Results aggregated: evaluation_results/${dataset}/summary.json"
        return 0
    else
        log_error "Aggregation failed"
        return 1
    fi
}
