from argparse import ArgumentParser
import os
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from analogy_creator import AnalogyCreator
import utils
from PIL import Image
import pickle as pkl
import torch
import numpy as np



parser = ArgumentParser()
parser.add_argument('--config',  dest='config', type=str, default='./config/parameter_estimation.yaml',
                    help='path to config file')

parser.add_argument('--inversion_subfolder',  dest='subfolder', type=str, default = 'noise',
                    help='inversion subfolder name')

parser.add_argument('--token_subfolder',  dest='token_subfolder', type=str, default = 'tokens',
                    help='Token inversion subfolder name')

parser.add_argument('--output_subfolder',  dest='out_subfolder', type=str, default = 'analogies',
                    help='Output subfolder name')

parser.add_argument('--triplet_file',  dest='triplet_file', type=str,
                    help='file with image paths')


parser.add_argument('--data_path',  dest='data_path', type=str, default = './dataset/data/',
                    help='root path to data')

args = parser.parse_args()
args = vars(args)


with open(args['triplet_file'], 'r') as f:
    file_lines = f.readlines()
    
    
clean_file_triplets = []
for line in file_lines:
    clean_line = line.replace('\n','').split(' ')
    pathA = utils.file_id2im_path(clean_line[0], data_path=args['data_path'],absolute=True)
    pathAprime = utils.file_id2im_path(clean_line[1], data_path=args['data_path'],absolute=True)
    pathB = utils.file_id2im_path(clean_line[2], data_path=args['data_path'],absolute=True)
    clean_file_triplets.append((pathA, pathAprime, pathB))


config = OmegaConf.load(f"{args['config']}")
config.args = args
# log_config = OmegaConf.load('./config/logging_config.yaml')
experiment_root = './results/'


subfolder = args['subfolder']
token_subfolder = args['token_subfolder']

print('Loading model...')
model = utils.prepare_default_model()
model = model.to(config.device)
print('Model loaded')

ddim_sampler = DDIMSampler(model)

export_path = os.path.join(experiment_root, 'analogy_results', args['out_subfolder'])

analogy_creator =  AnalogyCreator(config, ddim_sampler, subfolder, token_subfolder, export_path, data_path= args['data_path'])


analogy_config = OmegaConf.load(f"./config/analogy_params.yaml")
add_orig_row = analogy_config.add_orig_row 
scales = analogy_config.guidance_scales
steps = np.linspace(*analogy_config.analogy_strength)

analogy_func = lambda cA, cAprime, cB, st: cB + st * (cAprime - cA)

    
for triplet in clean_file_triplets:
    print(f'Processing triplet: {triplet}')
    if os.path.exists(os.path.join(export_path, utils.tuple2triplet_name(triplet))):
        print(f'This ({utils.tuple2triplet_name(triplet)}) analogy is precomputed... Skipping')
        continue

    if not utils.check_inversion_done(os.path.join(args['data_path'], triplet[2]), subfolder):
        print(f'Inversion for image {triplet[2]} not found... Skipping')
        print(f'p at : {os.path.join(args["data_path"], triplet[2], subfolder)}')
        continue
    if (not utils.check_inversion_done(os.path.join(args['data_path'], triplet[0]), token_subfolder) or \
        not utils.check_inversion_done(os.path.join(args['data_path'], triplet[1]), token_subfolder) or \
        not utils.check_inversion_done(os.path.join(args['data_path'], triplet[2]), subfolder)):
        print(f'Inversion not found... Skipping')
        continue

    analogy_creator.make_analogy(triplet, steps, scales, analogy_func = analogy_func)

    if add_orig_row:
        triplet_code = '_'.join([utils.extract_file_id_from_path(t) for t in triplet])
        grid = np.array(Image.open(os.path.join(export_path,
                        'grids',
                        f'{triplet_code}_analogy_grid.jpg' )))
        first_row = utils.join_images(triplet, out_PIL=False)
        n_pad = grid.shape[1] - first_row.shape[1]
        pad = np.zeros((first_row.shape[0], n_pad, 3))
        first_row = np.concatenate((first_row, pad), axis = 1)
        final_grid = np.uint8(np.concatenate((first_row, grid), axis = 0))
        Image.fromarray(final_grid).save(os.path.join(export_path,'grids',f'{triplet_code}_analogy_grid.jpg'))


