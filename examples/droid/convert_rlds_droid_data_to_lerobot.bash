#!/bin/bash

SCRIPT_PATH="/home/zhongzd/trzhang/repos/openpi-pf/examples/droid/convert_rlds_droid_data_to_lerobot.py"
INPUT_DIR="/home/zhongzd/trzhang/datasets/droid_100_raw"
MAPPING_FILE="/home/zhongzd/trzhang/datasets/droid_json/episode_id_to_path.json"
REPO_ID="droid_100_dataset_lerobot"
OUTPUT_ROOT="/home/zhongzd/trzhang/datasets/droid_100_lerobot"

PUSH_FLAG="" 
# PUSH_FLAG="--push-to-hub"

echo "Starting conversion..."
echo "Input: $INPUT_DIR"
echo "Mapping: $MAPPING_FILE"

python "$SCRIPT_PATH" \
    --input-dir "$INPUT_DIR" \
    --mapping-file "$MAPPING_FILE" \
    --repo-id "$REPO_ID" \
    --output-root "$OUTPUT_ROOT" \
    $PUSH_FLAG

echo "Conversion finished."
