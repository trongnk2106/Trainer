import os
import torch
from diffusers import StableDiffusionPipeline


model_normal = "runwayml/stable-diffusion-v1-5" 
model_xl = "stabilityai/stable-diffusion-xl-base-1.0"

# check path is exist
path_save_model = './pretrained_model'
if os.path.exists(path_save_model) is False:
    os.makedirs(path_save_model)
    print(f"Create folder: {path_save_model}")
else: 
    print(f"Folder exist: {path_save_model}")


# download the model
pipe = StableDiffusionPipeline.from_pretrained(model_normal, torch_dtype=torch.float16, cache_dir=path_save_model)
pipe = StableDiffusionPipeline.from_pretrained(model_xl, torch_dtype=torch.float16, cache_dir=path_save_model)