#!/bin/bash
# Colon Cancer Training Script for use INSIDE a Docker container on EKS

set -e # Exit on any error

echo "=========================================="
echo "Colon Cancer Cell Classification Pipeline"
echo "=========================================="

# --- Function Definitions ---

# Function to run training
run_training() {
    echo "Starting training..."
    python train.py \
        --train_csv /app/data/train.csv \
        --train_dir /app/data/train/ \
        --model_path /app/models/best_colon_cancer_model.pth \
        --config /app/outputs/hyperopt_results.json # Assumes config file might exist
}

# Function to generate training report
generate_report() {
    echo "Generating training report..."
    python -c "
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load training history
try:
    with open('/app/outputs/training_history.json', 'r') as f:
        history = json.load(f)
    
    # Create report
    report = {
        'training_completed': True,
        'best_val_accuracy': max(history['val_accs']),
        'final_train_accuracy': history['train_accs'][-1],
        'total_epochs': len(history['train_accs']),
        'best_epoch': history['val_accs'].index(max(history['val_accs'])) + 1
    }
    
    print('Training Report:')
    print(f'Best Validation Accuracy: {report[\"best_val_accuracy\"]:.2f}%')
    print(f'Final Training Accuracy: {report[\"final_train_accuracy\"]:.2f}%')
    print(f'Best Epoch: {report[\"best_epoch\"]}/{report[\"total_epochs\"]}')
    
    with open('/app/outputs/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
except FileNotFoundError:
    print('Training history not found. Please complete training first.')
"
}

# --- Argument Parsing ---
TRAIN=false
REPORT=false
ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            TRAIN=true
            shift
            ;;
        --report)
            REPORT=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --help)
            echo "Usage: [ --train | --report | --all | --help ]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$TRAIN" = false ] && [ "$REPORT" = false ] && [ "$ALL" = false ]; then
    echo "No action specified. Use --help for usage information."
    exit 1
fi

if [ "$ALL" = true ] || [ "$TRAIN" = true ]; then
    run_training
fi

if [ "$ALL" = true ] || [ "$REPORT" = true ]; then
    generate_report
fi

echo "=========================================="
echo "Pipeline execution completed!"
echo "=========================================="