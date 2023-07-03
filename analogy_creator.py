import os
import numpy as np
from PIL import Image

from ddim_invertor import DDIMInvertor
import utils

class AnalogyCreator():
    def __init__(self, config, ddim_sampler, inversion_subfolder, token_subfolder, output_root, data_path) -> None:
        self.config = config
        self.ddim_sampler = ddim_sampler
        self.subfolder = inversion_subfolder
        self.token_subfolder = token_subfolder
        self.output_root = output_root
        self.uc = self.ddim_sampler.model.get_learned_conditioning([''])
        self.data_path = data_path
            

    def fetch_cond_matrix(self, file_id):
        cond_out = utils.load_estimated_cond(file_id, token_subfolder=self.token_subfolder)
        cond_out = self.ddim_sampler.model.cond_stage_model.transformer(inputs_embeds = cond_out.unsqueeze(0))['last_hidden_state']
        return cond_out.to(self.config.device)

    def __load_B_noise(self, imB):
        fileid_B = utils.extract_file_id_from_path(imB)
        _,_,_, resB = utils.load_inversion_result_dict(fileid_B, self.subfolder, return_result_dict=True)
        return resB['estimated_input_noise'] 

    def make_analogy(self, triplet, steps, uc_scales, analogy_func = None, **analogy_func_kwargs):
        print(f'Make analogy inputs: {triplet}, {uc_scales}, {steps}')
        triplet_code = '_'.join([utils.extract_file_id_from_path(t) for t in triplet])
        noise = self.__load_B_noise(triplet[-1])

        cA = self.fetch_cond_matrix(utils.extract_file_id_from_path(triplet[0]))
        cAprime = self.fetch_cond_matrix(utils.extract_file_id_from_path(triplet[1]))

        cB = self.fetch_cond_matrix(utils.extract_file_id_from_path(triplet[2]))
        self.make_analogy_from_args(triplet_code, cA, cAprime, cB, noise, steps, uc_scales, analogy_func, **analogy_func_kwargs)


    def make_analogy_from_args(self, triplet_code, cA, cAprime, cB, noise, steps, uc_scales, analogy_func = None, **analogy_func_kwargs):
        
        os.makedirs(os.path.join(self.output_root, triplet_code), exist_ok=True)
        os.makedirs(os.path.join(self.output_root,'grids'), exist_ok=True) 
        if analogy_func is None:
            analogy_func = lambda cA, cAprime, cB, st: cB + st * (cAprime - cA)

        rows = []
        for sc in uc_scales:
            cols = []
            for st in steps:
                analogy_res,_ = self.ddim_sampler.sample(
                    self.config.ddim_steps,
                    1,
                    self.config.shape,
                    conditioning = analogy_func(cA, cAprime, cB, st, **analogy_func_kwargs),
                    eta=self.config.ddim_eta,
                    x_T=noise,
                    unconditional_guidance_scale=sc,
                    unconditional_conditioning=self.uc,
                )
                img = utils.save_latent_as_image(
                    self.ddim_sampler.model, 
                    analogy_res,
                    os.path.join(self.output_root, triplet_code, f'analogy_sc={sc}_shift_strength={st}.jpg'),
                    return_pil=True
                )
                cols.append(np.array(img))
            
            rows.append(np.concatenate(cols, axis = 1))
        grid = np.concatenate(rows, axis = 0)
        Image.fromarray(grid).save(os.path.join(self.output_root,
                        'grids',
                        f'{triplet_code}_analogy_grid.jpg' ))

