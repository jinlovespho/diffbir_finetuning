MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
dataset_path="./generated_data"
dataset_name="textocr"

DATA_ARGS="
  --dataset_name ${dataset_name} \
  --dataset_path ${dataset_path} \
  --resolution 512 \
"

MODEL_ARGS="
  --pretrained_model_name_or_path=${MODEL_NAME} \
"

TRAINING_ARGS="
  --seed 42 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --gradient_accumulation_steps=16 \
  --gradient_checkpointing \
  --checkpointing_steps 500 \
  --learning_rate 1e-5 \
  --max_train_steps=30000 \
  --train_batch_size 4 \

"
  # --max_train_samples 50 \

VAL_ARGS="
  --validation_epochs 1 \
  --val_batch_size 6 \
"

LOGGING_ARGS="
  --output_dir output_dir \
  --report_to wandb \
  --tracker_project_name DiffBIR_OCR_UNet\
  --wandb_exp_name pho_gpu1_textocr_UNet_lr1e-5_bs4_gradaccum16 \
"

ETC_ARGS="
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
"

CUDA_VISIBLE_DEVICES=1 accelerate launch diffusers/examples/text_to_image/train_text_to_image.py  ${DATA_ARGS} \
                                                                                                  ${MODEL_ARGS} \
                                                                                                  ${TRAINING_ARGS} \
                                                                                                  ${VAL_ARGS} \
                                                                                                  ${LOGGING_ARGS} \
                                                                                                  ${ETC_ARGS}

# CUDA_VISIBLE_DEVICES=3 accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
#   --pretrained_model_name_or_path=${MODEL_NAME} \
#   --dataset_name=${dataset_name} \
#   --resolution=512 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --enable_xformers_memory_efficient_attention \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="test_unet" \
#   --report_to wandb \
#   --wandb_proj_name DiffBIR_OCR_UNet\
#   --wandb_exp_name pho_gpu1_UNet_test \
#   --seed 42 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --max_train_steps=15000 \
#   --validation_epochs 1 \
#   --train_batch_size 3 \





# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5" 
# export dataset_name="lambdalabs/naruto-blip-captions"

# CUDA_VISIBLE_DEVICES=3 accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$dataset_name \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --enable_xformers_memory_efficient_attention \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="sd-naruto-model" \
#   --push_to_hub