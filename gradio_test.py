from diffusers import (
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetSceneTextErasingPipeline,
    )
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os
import h5py



os.environ["CUDA_VISIBLE_DEVICES"]="0"

pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained('/datadrive/Control_DiT_kavsir/')



pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(torch.device('cuda'))

pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(1)

# image = np.load("/home/ec2-user/tts2/BTCV/data/BTCV/train_npz/case0010_slice073.npz", mmap_mode='r')
# im = Image.fromarray(np.uint8(image['image']*255)).convert('RGB')


# f1 = h5py.File('/home/ec2-user/tts2/BTCV/data/BTCV/test_vol_h5/case0001.npy.h5','r')
# im = Image.fromarray(np.uint8(f1['image'][114]*255)).convert('RGB')

im = Image.open("/home/ec2-user/tts2/Kvasir-SEG/images/cju7ezs7g2mxm098787atbran.jpg").convert("RGB").resize((512, 512))

image = pipe(
    im,
    [im],
    num_inference_steps=30,
    generator=generator,
    controlnet_conditioning_scale=1.0,
    guidance_scale=0.1
).images[0]



image.save('resut.png')



# def inf(image, mask_image):
    
#     image = Image.fromarray(image).resize((512, 512))
#     mask_image = Image.fromarray(mask_image).resize((512, 512))

#     image = pipe(
#         image,
#         mask_image,
#         [mask_image],
#         num_inference_steps=20,
#         generator=generator,
#         controlnet_conditioning_scale=1.0,
#         guidance_scale=1.0
#     ).images[0]

#     return np.array(image)



# if __name__ == "__main__":

#     demo = gr.Interface(
#     inf, 
#     inputs=[gr.Image(), gr.Image()], 
#     outputs="image",
#     title="Scene Text Erasing, IIT-Jodhpur",
#     )
#     demo.launch(share=True)