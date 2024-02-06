# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=20001 fastchat/train/train_mem.py \
torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path mtgv/MobileLLaMA-2.7B-Base  \
    --data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --bf16 True \
    --output_dir output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 1800 \
    --save_total_limit 8 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none


## --data_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/LM_data/data/NLP/sft_data/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry_.json \
