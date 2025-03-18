export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

python -u inference.py \
--task sr \
--upscale 4 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner none \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4 \
--input /home/cvlab12/project/jaewon/text_restoration/data_gen/ocr/result/crop/image/256 \
--output results/lsdir_256 \
--device cuda --precision fp32