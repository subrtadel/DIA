import os
import utils
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument('--config',  dest='config', type=str, default='./config/parameter_estimation.yaml',
                    help='path to config file')

parser.add_argument('--inversion_subfolder',  dest='subfolder', type=str, default = 'noise',
                    help='inversion subfolder name')

parser.add_argument('--token_subfolder',  dest='token_subfolder', type=str, default = 'tokens',
                    help='Token inversion subfolder name')

parser.add_argument('--output_subfolder',  dest='out_subfolder', type=str, default = 'analogies_cond_shift_first_phrase',
                    help='Output subfolder name')

parser.add_argument('--triplet_file',  dest='triplet_file', type=str, default='triplets.csv',
                    help='file with triplets')


parser.add_argument('--data_path',  dest='data_path', type=str, default = './dataset/data/',
                    help='root path to data')

args = parser.parse_args()
args = vars(args)



with open(args['triplet_file'], 'r') as f:
    file_lines = f.readlines()
    
    
conditioning_inversion_names = []
noise_inversion_names = []
for line in file_lines:
    clean_line = line.replace('\n','').split(' ')
    A_name = utils.file_id2im_path(clean_line[0])
    Aprime_name = utils.file_id2im_path(clean_line[1])
    B_name = utils.file_id2im_path(clean_line[2])
    
    conditioning_inversion_names.extend([A_name ,Aprime_name, B_name])
    noise_inversion_names.append(B_name)


with open('tmp_clip_inversion.txt','w') as f:
    for fn in set(conditioning_inversion_names):
        f.write(f'{fn}\n')

with open('tmp_noise_inversion.txt','w') as f:
    for fn in set(noise_inversion_names):
        f.write(f'{fn}\n')


os.system(f'python estimate_CLIP_features.py --config {args["config"]} --subfolder {args["token_subfolder"]} --input_img tmp_clip_inversion.txt --data_path {args["data_path"]}')


os.system(f'python estimate_input_noise.py --config {args["config"]} --input_img tmp_noise_inversion.txt --token_subfolder {args["token_subfolder"]} --subfolder {args["subfolder"]} --data_path {args["data_path"]}')


os.remove('tmp_clip_inversion.txt')
os.remove('tmp_noise_inversion.txt')