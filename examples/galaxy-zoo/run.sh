#!/bin/bash

# Exit script if any command fails
set -e

# Activate conda environment
if ! conda activate codeagent; then
  echo "Failed to activate codeagent environment"
  exit 1
fi

# Define variables
OUTPUT_FILE="codeagent_run_$(date +%Y%m%d_%H%M%S).log"
JOB_ID_FILE="codeagent_job_id.txt"

# Run codeagent with parameters
nohup codeagent --dataset . \
  --num_ideas 20 \
  --coder_main_model "o3-mini" \
  --coder_edit_model "o3-mini" \
  --code_model "gemini/gemini-2.5-pro-exp-03-25" \
  --plan_model "gemini/gemini-2.5-pro-exp-03-25" \
  --timeout 20000 \
  --gpu_id 0 \
  --test_mode \
  --collaborative_rounds 5 > "$OUTPUT_FILE" 2>&1 &

# Save the process ID for future reference
echo $! > "$JOB_ID_FILE"
echo "CodeAgent job started with PID: $!"
echo "Output is being saved to: $OUTPUT_FILE"
