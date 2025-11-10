#!/bin/bash
# Training wrapper - ALWAYS activates CCM3 venv before running

# Activate CCM3 virtual environment
source CCM3/bin/activate

# Run the training with all arguments passed through
python Chandra_trainer.py "$@"
