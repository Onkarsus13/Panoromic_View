This repo is inharited form the huggingface diffusers its having more customized implimentation of controlnet, diT and Pixart-alpha


Installation

```
cd Panoromic_view
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas accelerate safetensors huggingface_hub
pip install -e .[all,dev,notebooks]
python3 setup.py install
```

To Run the ```test_pano.py ``` and ```test_controlnet.py``` install ```cog_sdxl```

installation for ```cog_sdxl```

```
git clone https://github.com/replicate/cog-sdxl cog_sdxl
```

Then Run the following command to get the inference


```
CUDA_VISIBLE_DEVICES=0 python3 test_controlnet.py

CUDA_VISIBLE_DEVICES=0 python3 test__sdxl_pano.py
```

You can find the visual results in the ```visual_results``` directory









