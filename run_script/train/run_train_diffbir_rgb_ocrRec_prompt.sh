
# single gpu
CUDA_VISIBLE_DEVICES=1 accelerate launch train.py   --config ./configs/train/train_diffbir_rgb_ocrRec_prompt.yaml \
                                                    --bridge_config bridge_config/Bridge/ICDAR15/R_50_poly.yaml \


# multi-gpu
# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --gpu_ids all  train.py  --config ./configs/train/train_diffbir_rgb_ocrRecLoss.yaml \
#                                                                       --bridge_config bridge_config/Bridge/ICDAR15/R_50_poly.yaml \


