#!/bin/bash

export HF_HOME="${HF_HOME:-/inspire/hdd/ziho/datasets}"
export VISION_TOWER_CACHE_DIR="${VISION_TOWER_CACHE_DIR:-/inspire/hdd/ziho/pretrained_models}"

max_frames_num=${max_frames_num:-64}
force_sample=${force_sample:-"False"}

output_suffix=${output_suffix:-""}

if [ $# -gt 0 ]; then
    MODEL_PATHS=("$@")
else
    MODEL_PATHS=(
        "ziHoHe/VidLaDA-8B"
    )
fi

OUTPUT_PATH=${OUTPUT_PATH:-"./work_dirs/vidlada_eval"}

gen_tasks="videomme,egoschema,mlvu_dev,mlvu_test,lvbench,video_mmmu,longvideobench_val_v,mvbench"
TASK_NAMES=${TASK_NAMES:-"$gen_tasks"}

TOTAL_GPUS=${TOTAL_GPUS:-$(nvidia-smi -L | wc -l)}
echo "Detected $TOTAL_GPUS GPUs. Running tasks sequentially using all GPUs."
IFS=',' read -ra TASKS <<< "$TASK_NAMES"

for model_path in "${MODEL_PATHS[@]}"; do
    MODEL=llava_llada_video
    MODEL_NAME=llava_vidlada
    CONV_TEMPLATE=llava_llada
    
    MODEL_PATH_LAST=$(basename "$model_path")
    CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$MODEL_PATH_LAST"
    CURRENT_OUTPUT_PATH=$CURRENT_OUTPUT_PATH${output_suffix}

    for task in "${TASKS[@]}"; do
        case $task in
            videomme|egoschema|egoschema_subset|mlvu_dev|mlvu_test|lvbench|video_mmmu|longvideobench_val_v)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
                extra_model_args="max_frames_num=$max_frames_num,video_fps=1,add_time_instruction=False,force_sample=$force_sample"
                ;;
            mvbench)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
                extra_model_args="max_frames_num=$max_frames_num,video_fps=1,add_time_instruction=True,force_sample=$force_sample"
                ;;
            *)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
                extra_model_args="max_frames_num=$max_frames_num,video_fps=1,add_time_instruction=True,force_sample=$force_sample"
                ;;
        esac

        LOG_FILE_NAME="${task}_${extra_model_args//,/_}.log"
        echo "----------------------------------------"
        echo "Starting evaluation: Model=$MODEL_PATH_LAST, Task=$task"
        echo "Output dir: $CURRENT_OUTPUT_PATH"
        mkdir -p $CURRENT_OUTPUT_PATH
        cp $0 $CURRENT_OUTPUT_PATH
    
        LOG_PATH=$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME
        
        accelerate launch --num_processes=$TOTAL_GPUS --main_process_port 29500 -m lmms_eval \
            --model "$MODEL" \
            ${GEN_KWARGS:+--gen_kwargs="$GEN_KWARGS"} \
            --model_args "pretrained=$model_path,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME,$extra_model_args" \
            --tasks "$task" \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "$task" \
            --output_path "$CURRENT_OUTPUT_PATH" \
            --use_cache "$CURRENT_OUTPUT_PATH" \
            2>&1 | tee -a $LOG_PATH

        echo "Finished task: $task"
    done
done

echo "All evaluations completed."
