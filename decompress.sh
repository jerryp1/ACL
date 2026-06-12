#!/bin/bash
# ./decompress.sh --input ./compressed \
#                 --base /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/model.safetensors \
#                 --output_dir ./restored
# ./decompress.sh --input ./compressed \
#                 --output_dir ./restored
PYTHON="${PYTHON:-$(command -v python3)}"
exec "${PYTHON}" "$(dirname "$0")/decompress_checkpoint.py" "$@"
