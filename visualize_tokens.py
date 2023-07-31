
from omegaconf import OmegaConf
import numpy as np
import os


from ldm.models.diffusion.ddim import DDIMSampler
import utils


def fetch_cond_matrix(file_id, ddim_sampler, config):
    cond_out = utils.load_estimated_cond(file_id, token_subfolder=token_subfolder)
    cond_out = ddim_sampler.model.cond_stage_model.transformer(inputs_embeds = cond_out.unsqueeze(0))['last_hidden_state']
    return cond_out.to(config.device)


path_to_data = './dataset/data/'
experiment_root = './results'

config = OmegaConf.load('./config/parameter_estimation.yaml')


print('Loading model...')
model = utils.prepare_default_model()
print('Model loaded')



token_subfolder = 'tokens'
subfolder = 'noise'


ddim_sampler = DDIMSampler(model)

# analogy_creator =  AnalogyCreator(config, ddim_sampler, subfolder, token_subfolder, os.path.join(experiment_root, 'analogy_results', out_subfolder))
file_id_A = '00009'
file_id_A_prime = '00008'
file_id_B = '00010'



cA = fetch_cond_matrix(file_id_A, ddim_sampler, config)
cAprime = fetch_cond_matrix(file_id_A_prime, ddim_sampler, config)
cB = fetch_cond_matrix(file_id_B, ddim_sampler, config)


os.makedirs(os.path.join(experiment_root,'token_visualization'),exist_ok=True)
def gen_random_samples(cond, file_id):
    tokens_,_ = ddim_sampler.sample(
                    config.ddim_steps,
                    8,
                    config.shape,
                    conditioning = cond.expand(8,-1,-1),
                    eta=config.ddim_eta,
                    unconditional_guidance_scale=1.,
                    unconditional_conditioning=ddim_sampler.model.get_learned_conditioning(['']).expand(8,-1,-1),
                )
    utils.save_latent_as_image(
        ddim_sampler.model, 
        tokens_,
        os.path.join(experiment_root,'token_visualization',f'{file_id}.jpg')
    )

gen_random_samples(cA, file_id_A)
gen_random_samples(cAprime, file_id_A_prime)
gen_random_samples(cB, file_id_B)
