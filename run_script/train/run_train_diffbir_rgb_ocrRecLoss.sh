
# single gpu
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py   --config ./configs/train/train_diffbir_rgb_ocrRecLoss.yaml \
                                                    --bridge_config bridge_config/Bridge/ICDAR15/R_50_poly.yaml \


# multi-gpu
# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --gpu_ids all  train.py  --config ./configs/train/train_diffbir_rgb_ocrRecLoss.yaml \
#                                                                       --bridge_config bridge_config/Bridge/ICDAR15/R_50_poly.yaml \


