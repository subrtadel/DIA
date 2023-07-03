from argparse import ArgumentParser
from omegaconf import OmegaConf
from PIL import Image
import pickle as pkl
import torch
import os

from ddim_invertor import DDIMInvertor
import utils


parser = ArgumentParser()
parser.add_argument('--config',  dest='config', type=str, default='./config/noise_estimation.yaml',
                    help='path to config file')

parser.add_argument('--input_img',  dest='input_img', type=str, default = None,
                    help='path to image or text files with image names')

parser.add_argument('--subfolder',  dest='subfolder', type=str, default = 'noise',
                    help='subfolder name')

parser.add_argument('--token_subfolder',  dest='token_subfolder', type=str, default = 'tokens',
                    help='token subfolder name')


parser.add_argument('--data_path',  dest='data_path', type=str, default = '/home/subrtade/analogies/dataset/data/',
                    help='root path to data')

args = parser.parse_args()
args = vars(args)

assert os.path.isfile(args['input_img']), '--input_img is not a file'

if args['input_img'].endswith('.txt'):
    with open(args['img_img'], 'r') as f:
        file_lines = f.readlines()
    clean_file_lines = [os.path.join(args['data_path'],x.replace('\n','')) for x in file_lines]
elif args['input_img'].endswith(('.png','.jpeg','.jpg')):
    clean_file_lines = [args['input_img']]

config = OmegaConf.load(f"{args['config']}")
config.args = args
config.token_subfolder = args['token_subfolder']


print('Loading model...')
model = utils.prepare_default_model()
model = model.to(config.device)
print('Model loaded')

    
invertor = DDIMInvertor(config, model)

    
for file_name in clean_file_lines:
    print(f'Processing file: {file_name}')
    if not os.path.exists(file_name):
        print(f'Path {file_name} does not exist. Skipping')
        continue
    # load & prepare image
    file_id = utils.extract_file_id_from_path(file_name)
    export_path = os.path.join(config.path2save_prefix, file_id, args['subfolder'])
    if os.path.exists(os.path.join(export_path, 'results.pkl')):
        print(f'The inversion for {file_id} seems to be done already. Skipping...')
        continue
    os.makedirs(export_path, exist_ok=True)

    
    print('Performing inversion...')
    outputs = invertor.perform_inversion(file_name, cond = None, init_noise_init = None, loss_weights= {'latents': 1. , 'pixels':1.} )


    outputs['token_subfolder'] = args['token_subfolder']

    if config.save_reconstruction:
        img = utils.load_pil(file_name)
        img.save(os.path.join(export_path,f'target.png'))

        rec_img_torch = utils.latent2img(model, outputs['reconstruction'])
        rec_img_pil = utils.torch2pil(rec_img_torch)[0]
        rec_img_pil.save(os.path.join(export_path, 'reconstruction.png'))


    utils.save_results2pickle(export_path, outputs)


