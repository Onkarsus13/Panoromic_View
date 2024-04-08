import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")

pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")

text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
embhandler.load_embeddings(embedding_path)
prompt="hdri view, Create a panoramic view of a modern, sunlit kitchen, stretching across the image. On the left, include a stainless steel refrigerator beside tall wooden cabinets, in the style of <s0><s1>"
images = pipe(
    prompt,
    num_inference_steps=40,
    cross_attention_kwargs={"scale": 1.0},
    height=256,
    width=512,
    guidance_scale=0.0,
).images
#your output image
images[0].save('visual_results/pano.png')
