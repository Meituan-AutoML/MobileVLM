#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}

declare -A LR_CONF
LR_CONF=([mobilevlm1.7b]=2e-5 [mobilevlm3b]=4e-5)

ARCH=$1
TASK=$2
case ${TASK} in
    "pretrain-finetune-test")
        cd ${WORK_DIR}
        OUTPUT_DIR=${WORK_DIR}/outputs/${ARCH}_$(date +"%Y%m%d_%H%M%S")
        mkdir -p ${OUTPUT_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        bash run.sh ${ARCH} pretrain ${LANGUAGE_MODEL} ${VISION_MODEL} ${OUTPUT_DIR}
        bash run.sh ${ARCH} finetune ${LANGUAGE_MODEL} ${VISION_MODEL} ${OUTPUT_DIR}
        bash run.sh ${ARCH} test ${OUTPUT_DIR}/mobilevlm-2.finetune
    ;;
    "pretrain")
        echo ">>> Start Feature-Alignment Pretrain ..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm-1.pretrain
        mkdir -p ${OUTPUT_DIR_PT}
        deepspeed mobilevlm/train/train_mem.py \
            --deepspeed scripts/deepspeed/zero2.json \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --version plain \
            --data_path data/pretrain_data/blip_laion_cc_sbu_558k.json \
            --image_folder data/pretrain_data/images \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --mm_projector_type ldpnet \
            --tune_mm_mlp_adapter True \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_PT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 24000 \
            --save_total_limit 1 \
            --learning_rate 1e-3 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to none \
            2>&1 | tee ${OUTPUT_DIR_PT}/log.txt &&
        echo "Done."
    ;;
    "finetune")
        echo ">>> Start Visual-Instruction Tuning ..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm-1.pretrain
        OUTPUT_DIR_FT=${OUTPUT_DIR}/mobilevlm-2.finetune
        mkdir -p ${OUTPUT_DIR_FT}
        deepspeed mobilevlm/train/train_mem.py \
            --deepspeed scripts/deepspeed/zero3.json \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --version v1 \
            --data_path data/finetune_data/llava_v1_5_mix665k.json \
            --image_folder data/finetune_data \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --pretrain_mm_mlp_adapter ${OUTPUT_DIR_PT}/mm_projector.bin \
            --mm_projector_type ldpnet \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_FT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --learning_rate ${LR_CONF[${ARCH}]} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to none \
            2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
        echo "Done."
    ;;
    "test")
        echo ">>> Start Evaluation ..."
        cd ${WORK_DIR}
        OUTPUT_DIR=$3
        bash scripts/benchmark.sh ${OUTPUT_DIR}
    ;;
    *)
        echo "error with ${DATASET_ID}"
esac
