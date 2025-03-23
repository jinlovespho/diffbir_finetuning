MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
dataset_name="lambdalabs/naruto-blip-captions"

CUDA_VISIBLE_DEVICES=0 accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --dataset_name=${dataset_name} \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="test_sd-naruto-model" \
  --report_to tensorboard \
  --wandb_proj_name DiffBIR_OCR_UNet\
  --wandb_exp_name pho_gpu1_UNet_test \
  --seed 42 \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs 10 \
  --train_batch_size 3 \




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