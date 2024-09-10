CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ./saves/llama3-8b/lora/sft_2/ \
    --template llama3 \
    --finetuning_type lora
