from argparse import ArgumentParser
from omegaconf import OmegaConf
from PIL import Image
import pickle as pkl
import numpy as np 
import sys
import os
sys.path.append('/home/subrtade/analogies/DiffusionImageAnalogies/stable-diffusion/')

from ddim_invertor import DDIMInvertor
import utils



parser = ArgumentParser()
parser.add_argument('--config',  dest='config', type=str, default='./config/parameter_estimation.yaml',
                    help='path to config file')
parser.add_argument('--input_img',  dest='input_img', type=str, required = True,
                    help='path to image or text files with image names')

parser.add_argument('--subfolder',  dest='subfolder', type=str, default = 'tokens',
                    help='subfolder name')


parser.add_argument('--data_path',  dest='data_path', type=str, default = './dataset/data/',
                    help='root path to data')

parser.add_argument('--regenerate_tokens',  dest='regenerate', action='store_true',
                    help='Will regenerate images with random noise and the output conditioning')

args = parser.parse_args()
args = vars(args)

assert os.path.isfile(args['input_img']), '--input_img is not a file'

if args['input_img'].endswith('.txt'):
    with open(args['input_img'], 'r') as f:
        file_lines = f.readlines()
    clean_file_lines = [os.path.join(args['data_path'],x.replace('\n','')) for x in file_lines]
elif args['input_img'].endswith(('.png','.jpeg','.jpg')):
    clean_file_lines = [args['input_img']]

config = OmegaConf.load(f"{args['config']}")
config.args = args


print('Loading model...')
model = utils.prepare_default_model()
model = model.to(config.device)
print('Model loaded')


invertor = DDIMInvertor(config, model)

for file_path in clean_file_lines:
    if not os.path.exists(file_path):
        print(f'Path {file_path} does not exist. Skipping')
        continue

    file_id = utils.extract_file_id_from_path(file_path)
    if os.path.exists(os.path.join(config.path2save_prefix, file_id, args['subfolder'],'results.pkl')):
        print(f'Inversion for file_id {file_id} is already done... Skipping')
        continue

    output = invertor.perform_cond_inversion_individual_timesteps(file_path, None, optimize_tokens=True)


    export_path = os.path.join(config.path2save_prefix, file_id, args['subfolder'])
    
    utils.save_results2pickle(export_path, output)
    # os.makedirs(export_path, exist_ok=True)

    # with open(os.path.join(export_path, 'results.pkl') ,'wb') as f:
    #     pkl.dump(output, f)

    if args["regenerate"]:
        c_ = model.cond_stage_model.transformer(inputs_embeds = output['estimated_conditioning'].unsqueeze(0))['last_hidden_state']
        res, _ = invertor.ddim_sampler.sample(config.ddim_steps,
                    config.conditioning_optimization.batch_size,
                    config.shape,
                    conditioning=c_.expand(config.conditioning_optimization.batch_size, -1,-1),
                    eta=0.,
                    unconditional_guidance_scale=5.,
                    unconditional_conditioning=invertor.uc.expand(config.conditioning_optimization.batch_size, -1,-1))



        img = utils.save_latent_as_image(model, res, os.path.join(export_path,'token_regeneration.png'),return_pil=True)
        orig = np.array(Image.open(file_path).convert("RGB"))
        row = np.concatenate((orig,np.zeros((orig.shape[0], 20,3)),np.array(img)), axis = 1).astype(np.uint8)
        Image.fromarray(row).save(os.path.join(export_path,'token_regeneration_with_ref.png'))

    del output

