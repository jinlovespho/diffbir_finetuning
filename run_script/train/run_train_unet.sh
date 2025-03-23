
# single gpu
CUDA_VISIBLE_DEVICES=3 accelerate launch train_unet.py      --config ./configs/train/train_unet.yaml \
                                                            --bridge_config bridge_config/Bridge/ICDAR15/R_50_poly.yaml \

