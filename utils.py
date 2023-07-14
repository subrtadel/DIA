from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
import numpy as np
import pickle as pkl
import torch
import fnmatch
import PIL
import gc
import os


from ldm.util import instantiate_from_config
from modified_clip_transformers import ModifiedCLIPTextModel
import importlib

# importlib.import_module("/home/subrtade/analogies/DiffusionImageAnalogies/stable-diffusion")

######################################################################## Model prep



# taken from the stable-diffusion project script txt2img.py
def load_model_from_config(config, ckpt, verbose=False, device='cuda'):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def prepare_default_model(default_seed = 42):
    default_config_path = "/home/subrtade/analogies/DiffusionImageAnalogies/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    default_ckpt_path = "/home/subrtade/analogies/DiffusionImageAnalogies/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"

    seed_everything(default_seed)

    config = OmegaConf.load(f"{default_config_path}")
    model = load_model_from_config(config, default_ckpt_path, True)

    del model.cond_stage_model.transformer
    print(f'GC COLLECT RETURN VALUE: {gc.collect()}')


    model.cond_stage_model.transformer = ModifiedCLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(model.device)


    return model


#######################################################################
#  Data prep 

def extract_file_id_from_path(file_name):
    return os.path.basename(file_name).split('.')[0]

def load_all_image_names(path = './dataset/data/', suffixes = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG']):
    all_image_files = os.listdir(path)
    extracted_files = []
    for suf in suffixes:
        extracted_files += fnmatch.filter(all_image_files , f'*.{suf}')

    return extracted_files


def file_id2im_path(file_id, data_path = './dataset/data', absolute=False):
    if not file_id.endswith(('png', 'jpg', 'jpeg', 'JPG', 'JPEG')):    
        image_names = load_all_image_names(path=data_path)
        image_name = fnmatch.filter(image_names, f'{file_id}.*')[0]
    else:
        image_name = file_id
    if absolute:
        return os.path.join(data_path, image_name)
    return image_name

    
def extract_triplet_from_tuple(tuple_):
    fids = [extract_file_id_from_path(pth) for pth in tuple_]
    return fids

def tuple2triplet_name(triplet_tuple):
    file_ids = extract_triplet_from_tuple(triplet_tuple)
    return '_'.join(file_ids)


def join_images(list_of_image_paths, dim=1, path_prefix = '',out_PIL = True):
    """Given list of image paths, the function puts the images side by side in 'dim'.

    Args:
        list_of_image_paths (list): List that contains paths to the images
        dim (int, optional): In which dimension are the images joined. Defaults to 1.
        path_prefix (str, optional): Path to the images. Defaults to ''.
        out_PIL (bool, optional): The output is PIL image if True, otherwise the ouptut is np.ndarray. Defaults to True.

    Returns:
        _type_: _description_
    """    
    imgs = []
    for im_name in list_of_image_paths:
        img = np.array(PIL.Image.open(os.path.join(path_prefix,im_name)))
        if len(img.shape) == 2:
            img = np.stack((img,img,img), axis = -1)
        if img.shape[-1] == 4:
            img = img[:,:,:3]
        imgs.append(img)
    return join_array_of_np_images(imgs, dim, out_PIL)

def join_array_of_np_images(array_of_imgs, dim = 1, out_PIL = True):
    if out_PIL:
        return PIL.Image.fromarray(np.concatenate(array_of_imgs, axis = dim))
    return np.concatenate(array_of_imgs, axis = dim)
    



def img2latent(model, img_torch):
    return model.get_first_stage_encoding(model.encode_first_stage(img_torch.to(model.first_stage_model.device)))

def latent2img(model, latent):
    images = model.decode_first_stage(latent.to(model.first_stage_model.device))
    return images

def load_pil(img):
    return PIL.Image.open(img).convert('RGB')


def pil2torch(pilimg, to_range = True, device = 'cuda:0'):
    w, h = pilimg.size
    w, h = w - w%32, h - h%32
    pilimg.resize((w,h), resample=PIL.Image.LANCZOS)
    im_np = np.array(pilimg).astype(np.float32) / 255.
    im_np = im_np[np.newaxis].transpose((0, 3, 1, 2))
    im_torch = torch.from_numpy(im_np).to(device)
    if to_range:
        im_torch = 2*im_torch - 1
    return im_torch

def pil2torch_batch(list_of_ims, to_range=True, device= 'cuda:0'):
    batch = []
    for b in range(len(list_of_ims)):
        batch.append(pil2torch(list_of_ims[b], to_range, device))
    return torch.cat(batch, dim=0)

def torch2pil(images, from_range = True):
    if from_range:
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    pil_images = []
    b = images.shape[0]
    for i in range(b):
        img_np = np.array(images[i].detach().cpu())
        img_np = np.uint8(img_np.transpose((1,2,0)) * 255)
        img_pil = PIL.Image.fromarray(img_np)
        pil_images.append(img_pil)
    return pil_images


def save_latent_as_image(model, latent, path, return_pil=False):
    """Generates the output image from given latent and saves it to path.

    Args:
        model (_type_): stable diffusion.
        latent (_type_): Latent of the image.
        path (_type_): Path to save the image.
    """
    rec_img_torch = latent2img(model, latent)
    rec_img_pil = torch2pil(torch.cat(list(rec_img_torch), dim = -1).unsqueeze(0))[0]
    rec_img_pil.save(path)
    if return_pil:
        return rec_img_pil

#######################################################################
#  Optimization utils 

def pixel_space_loss(model, latent1, real_image, loss_fn):
    """computes loss loss_fn in pixel space 

    Args:
        model (_type_): stable diffusion model
        latent1 (_type_): latent of the generated image
        real_image (_type_): target image
        loss_fn (_type_): torch functional loss

    Returns:
        _type_: Value of loss_fn between real_image and the image generated from latent1.
    """
    image1 = model.differentiable_decode_first_stage(latent1.to(model.first_stage_model.device))
    return loss_fn(image1, real_image.to(image1.device))


#######################################################################
#  Results manipulation

def load_estimated_cond(file_id, token_subfolder = 'tokens', inversion_path_root = './results/experiments/inversion/' ):
    if not os.path.exists(os.path.join(inversion_path_root, f'{file_id}/{token_subfolder}/results.pkl')):
        return None
    with open(os.path.join(inversion_path_root,f'{file_id}/{token_subfolder}/results.pkl'),'rb') as f:
        results = pkl.load(f)
    return results['estimated_conditioning']



def load_inversion_result_dict(file_id, subfolder, return_result_dict = False, inversion_root_folder='./results/experiments/inversion/'):
    """Loads the results of inversion for given file_id and experiment.

    Args:
        file_id (str (ex. 000001)): File id of the inverted image.
        subfolder (str): Name of the inversion experiment.
        return_result_dict (bool, optional): If yes returns the whole result dict. Defaults to False.

    Returns:
        _type_: collection of (noise, conditioning matrix, unconditional guidance scale, [result dict])
    """
    assert os.path.exists(os.path.join(inversion_root_folder, file_id, subfolder,'results.pkl')) , f'This ({file_id}/{subfolder}) experiment does not exist.'

    with open(os.path.join(inversion_root_folder, file_id, subfolder,'results.pkl'), 'rb') as f:
        results = pkl.load(f)

    noise = results['estimated_input_noise'] if 'estimated_input_noise' in results.keys() else None
    cond = results['estimated_conditioning'] if 'estimated_conditioning' in results.keys() else None
    cond_scale = results['guidance_scale'] if 'guidance_scale' in results.keys() else None
    
    output = (noise, cond, cond_scale)
    if return_result_dict:
        output = (*output, results)
    return output



def check_inversion_done(path_to_image_or_file_id, subfolder, inversion_root_folder = "./results/experiments/inversion/"):
    if path_to_image_or_file_id.endswith(('.jpg','.png','.jpeg', 'JPG', 'JPEG')):
        file_id = extract_file_id_from_path(path_to_image_or_file_id)
    else:
        file_id = path_to_image_or_file_id
    print(f'Checking: {os.path.join(inversion_root_folder, file_id, subfolder,"results.pkl")}')
    return os.path.exists(os.path.join(inversion_root_folder, file_id, subfolder,'results.pkl'))

#######################################################################
#  Others


def save_results2pickle(path2save, results):
    os.makedirs(path2save, exist_ok=True)
    with open(os.path.join(path2save, 'results.pkl') ,'wb') as f:
        pkl.dump(results, f)



def check_and_run_inversion(model, file_id, subfolder, config, tokens = True):
    if not check_inversion_done(file_id, subfolder):
        from ddim_invertor import DDIMInvertor
        invertor = DDIMInvertor(config, model)
        if tokens:
            output = invertor.perform_cond_inversion_individual_timesteps(file_id2im_path(file_id), None, optimize_tokens=True)
        else:
            output = invertor.perform_inversion(file_id2im_path(file_id), None, init_noise_init = None, loss_weights= {'latents': 1. , 'pixels':1.} )

        export_path = os.path.join(config.path2save_prefix, file_id, subfolder)
        save_results2pickle(export_path, output)
    print(f'Inversion done')