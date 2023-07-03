# DiffusionImageAnalogies
![DIA_Teaser](https://github.com/subrtadel/DIA/assets/129282989/5f11b34d-9f49-47a2-b90d-60ee36ebc3bc)
This is the official repository for the Diffusion Image Analogies paper.


## Installation

1. Clone the repo
   ```sh
   git clone --recurse-submodules https://github.com/subrtadel/DIA.git
   ```
2. Create environment 
  ```
  conda install python=3.8.5 pip=20.3 cudatoolkit=11.3 pytorch=1.11.0 torchvision=0.12.0 numpy=1.19.2 -c pytorch -c conda-forge -c defaults
  ```
3. Install packages
   ```sh
   pip install -r requirements.txt
   cd ./DIA/stable-diffusion/
   pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
   pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
   pip install -e .
   ```
4. Download the [sd-v1-4.ckpt model](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) put it into the correct folder
  ```
  mkdir -p ./DIA/stable-diffusion/models/ldm/stable-diffusion-v1/

  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

##Usage

##Citation
