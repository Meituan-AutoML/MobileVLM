#!/bin/bash

GPUS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPUS"
CHUNKS=${#GPULIST[@]}

MODEL_LOADER=$1
MODEL_DIR=$2
CONV_MODE=$3
SPLIT_NME=$4
DATA_ROOT=$5
SAVE_PATH=$6/${SPLIT_NME}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ${MODEL_LOADER} \
        --model-path ${MODEL_DIR} \
        --question-file ${DATA_ROOT}/${SPLIT_NME}.jsonl \
        --image-folder ${DATA_ROOT}/val2014 \
        --answers-file ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV_MODE} &
done

wait

RESULT_FILE=${SAVE_PATH}/predictions.jsonl
> "${RESULT_FILE}"  # Clear out the output file if it exists
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl >> "${RESULT_FILE}"  # Loop through the indices and concatenate each file
    rm ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl
done

python ${DATA_ROOT}/eval.py \
    --question-file ${DATA_ROOT}/${SPLIT_NME}.jsonl \
    --annotation-dir ${DATA_ROOT}/coco \
    --result-file ${SAVE_PATH}/predictions.jsonl
