DEBUG=0

export OMP_NUM_THREADS=8
export VISION_TOWER_CACHE_DIR="${VISION_TOWER_CACHE_DIR:-/inspire/hdd/ziho/pretrained_models}"

num_node=${1:-1}
gpu_num_per_node=$(nvidia-smi -L | wc -l)
gpu_num=$gpu_num_per_node

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="${LLM_VERSION:-GSAI-ML/LLaDA-V}"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-google/siglip2-so400m-patch14-384}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

VIDEO_FOLDER="${VIDEO_FOLDER:-/inspire/hdd/ziho/datasets/llava-video-178k-data/v1/data}"

# batch
TARGET_EFFECTIVE_BATCH_SIZE=64
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo $gpu_info
if echo "$gpu_info" | grep -q 'H200'; then
    PER_DEVICE_BATCH_SIZE=2
else
    PER_DEVICE_BATCH_SIZE=1
fi

TOTAL_GPU_NUM=$((num_node * gpu_num))
GRAD_ACCUM_STEPS=$((TARGET_EFFECTIVE_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * TOTAL_GPU_NUM)))

if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi

echo "--- Batch Size Calculation ---"
echo "Target Effective Batch Size: ${TARGET_EFFECTIVE_BATCH_SIZE}"
echo "Total GPU Count: ${TOTAL_GPU_NUM}"
echo "Batch Size per GPU: ${PER_DEVICE_BATCH_SIZE}"
echo "Calculated Gradient Accumulation Steps: ${GRAD_ACCUM_STEPS}"
echo "------------------------------"

############### Finetune ################

PROMPT_VERSION="llava_llada"

DATA_YAML=scripts/data/exp.yaml
DATA_VERSION=$(basename "$DATA_YAML" .yaml)
BASE_RUN_NAME="llada-video-ft-nf64-bs64-${DATA_VERSION}"

OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/$BASE_RUN_NAME}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

touch $OUTPUT_DIR/num_node_${num_node}.txt

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path $DATA_YAML \
    --video_folder "${VIDEO_FOLDER}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --video_fps 1 \
    --frames_upbound 64 \
    --add_time_instruction True \
    --force_sample True \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    2>&1 | tee $OUTPUT_DIR/log_rank_$RANK.txt
