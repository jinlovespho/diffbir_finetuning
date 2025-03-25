from diffusers import DiffusionPipeline
import torch

# pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path='./ddpm-ema-flowers-64/checkpoint-500').to("cuda")
# image = pipeline().images[0]


from diffusers import UNet2DConditionModel

config = UNet2DConditionModel.load_config("CompVis/stable-diffusion-v1-4", subfolder="unet")
model1 = UNet2DConditionModel.from_config(config)

model2 = UNet2DConditionModel()

p1 = sum(i.numel() for i in model1.parameters())
p2 = sum(i.numel() for i in model2.parameters())
breakpoint()