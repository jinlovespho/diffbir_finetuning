CUDA_VISIBLE_DEVICES=2 accelerate launch train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --output_dir="ddpm-ema-flowers-64" \
  --mixed_precision="fp16" \
  --save_images_epochs 1 \
  --logger wandb \