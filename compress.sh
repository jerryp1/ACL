#!/bin/bash
# ./compress.sh --input /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/checkpoint-10/model.safetensors \
#               --base /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/model.safetensors \
#               --output_dir ./compressed
# ./compress.sh --input /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/checkpoint-10/global_step10/mp_rank_00_model_states.pt \
#               --base /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/model.safetensors \
#               --output_dir ./compressed
# ./compress.sh --input /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/checkpoint-20/global_step20/ \
#               --base_dir /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/checkpoint-10/global_step10/ \
#               --output_dir ./compressed
# ./compress.sh --input /data/oss_bucket_0/finetuning/ckpt/qwen35-0.8b-full-sft-v3/checkpoint-10/model.safetensors \
#               --output_dir ./compressed
PYTHON="${PYTHON:-$(command -v python3)}"
exec "${PYTHON}" "$(dirname "$0")/compress_checkpoint.py" "$@"
