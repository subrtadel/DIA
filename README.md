# Diffusion Image Analogies
<div align='center'>
<a src ="https://cmp.felk.cvut.cz/~subrtade/">Adéla Šubrtová</a>,
<a src ="https://research.adobe.com/person/michal-lukac/">Michal Lukáč</a>,  
<a src ="https://cmp.felk.cvut.cz/~cechj/">Jan Čech</a>,  
David Futschik,  
<a src ="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a>,  
<a src ="https://dcgi.fel.cvut.cz/home/sykorad/">Daniel Sýkora</a>,  
</div>

![DIA_Teaser](https://github.com/subrtadel/DIA/assets/129282989/4e5ab11d-851a-4d9a-a6f8-d3769e994e33)
This is the official repository for the Diffusion Image Analogies paper published at the SIGGRAPH 2023 Conference Proceedings.

***

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
   pip install -e .
   ```
4. Download the [sd-v1-4.ckpt model](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and put it into correct folder
    ```
    mkdir -p ./models/ldm/stable-diffusion-v1/

    ```
5. Install [Image Magick](https://imagemagick.org).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***


## Usage

1. Upload images into `./dataset/raw_data/` folder.
</br>
2. Run `process_new_data.py`. The images are assigned `file_id`s in a `%05d` format.
</br>
3. Define the triplets in a `.csv` file. Refer to the images by their `file_id`. 
    Example file is `triplets.csv`. First column specifies `A` inputs, second `A'` and the third `B` inputs. Either with of without filename suffixes is fine.
</br>
4. Run the `precompute_noises_and_conditionings.py` script. This may take a while.
    ``` python
    python precompute_noises_and_conditionings.py \
        --config ./config/parameter_estimation.yaml \
        --inversion_subfolder noise \
        --token_subfolder tokens \ 
        --triplet_file triplets.csv \
        --data_path ./dataset/data/
    ```

</br>

5. Check the `./config/analogy_params.yaml`.
</br>

6. Run the `do_analogies.py` script. 
    ``` python do_analogies.py
    python do_analogies.py \
        --config ./config/parameter_estimation.yaml \
        --inversion_subfolder noise \
        --token_subfolder tokens \ 
        --output_subfolder analogies \
        --triplet_file triplets.csv \
        --data_path ./dataset/data/
    ```



***

## BibTeX

    @inproceedings{Subrtova2023DIA,
        title = {Diffusion Image Analogies},
        author = {A. \v{S}ubrtov\'{a} and M. Luk\'{a}\v{c} and J. \v{C}ech and D. Futschik and E. Shechtman  and D. S\'{y}kora},
        booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
        year = {2023}
      }
