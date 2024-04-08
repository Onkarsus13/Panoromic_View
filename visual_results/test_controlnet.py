from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

# depth_estimator = pipeline('depth-estimation')

# image = load_image("/home/ec2-user/tts2/DiffCTSeg/pano_depth.png")

# image = depth_estimator(image)['depth']
# image = np.array(image)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# image = Image.fromarray(image)

image = Image.open('./pano_depth.png')

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

prompt = "Generate a panoramic image kitchen in which freeze is there"
negative_prompt = "monochrome, lowres, low quality"
image = pipe(prompt, image, negative_prompt= negative_prompt, num_inference_steps=33, height=256, width=1024).images[0]

image.save('./gen_control_depth_pano_p5.png')