from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from ldm.modules.diffusionmodules.util import noise_like
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import torch
import gc

from modified_clip_transformers import ModifiedCLIPTextModel
import utils


class DDIMInvertor():
    def __init__(self, config, model, tokenizer=None) -> None:
        self.config = config
        self.ddim_sampler = DDIMSampler(model)
        self.ddim_sampler.make_schedule(self.config.ddim_steps, ddim_eta=self.config.ddim_eta, verbose=False)
        self.uc = self.ddim_sampler.model.get_learned_conditioning([''])
        self.tokenizer = tokenizer


    def __sample_differentiable(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      timesteps=None, unconditional_guidance_scale=1., 
                      unconditional_conditioning=None,
                      ):
        b = cond.shape[0]              
        if x_T is None:
            img = torch.randn(shape, device=self.config.device)
        else:
            img = x_T
        if timesteps is None:
            timesteps = self.ddim_sampler.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_sampler.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_sampler.ddim_timesteps.shape[0], 1) * self.ddim_sampler.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_sampler.ddim_timesteps[:subset_end]

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.config.device, dtype=torch.long)


            outs = self.__step_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
        return img


    def __step_ddim(self, x, c, t, index, use_original_steps=False,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.ddim_sampler.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.ddim_sampler.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        
        alphas = self.ddim_sampler.model.alphas_cumprod if use_original_steps else self.ddim_sampler.ddim_alphas
        alphas_prev = self.ddim_sampler.model.alphas_cumprod_prev if use_original_steps else self.ddim_sampler.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sampler.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sampler.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sampler.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sampler.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, False)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


    def perform_inversion(self, image, cond, init_noise_init = None, loss_weights = {'latents': 1. , 'pixels':1.} ):
        if cond is None:
            with torch.no_grad():
                cond_out = utils.load_estimated_cond(utils.extract_file_id_from_path(image), token_subfolder=self.config.token_subfolder)
                cond = self.__tokens2conditioning(cond_out)
        
        target_img = utils.load_pil(image)
        target_img = target_img.resize((self.config.shape[-2] * self.config.f, self.config.shape[-1] * self.config.f))
        target_img = utils.pil2torch(target_img)
        target_latent = utils.img2latent(self.ddim_sampler.model, target_img)
        target_latent = target_latent.to(self.ddim_sampler.model.device)
        target_img = target_img.to(self.ddim_sampler.model.device)


        if init_noise_init is None:
            alpha_t = torch.tensor([self.ddim_sampler.ddim_alphas[-1]]).cuda()
            init_noise =  torch.sqrt(alpha_t) * target_latent + torch.sqrt(1. - alpha_t) * torch.randn_like(target_latent).to(target_latent.device)

        uc_scale = self.config.noise_optimization.uncond_guidance_scale

        init_noise.requires_grad = True

        lbfgs = torch.optim.LBFGS(params = [init_noise], lr = self.config.noise_optimization.lr)
        loss_fn = torch.nn.functional.mse_loss

        shape = [self.config.noise_optimization.batch_size, * self.config.shape]


        progress = {'loss':[]}
        progress['noise'] = []

        pbar = tqdm(range(self.config.noise_optimization.opt_iters))
        for i in pbar:
            def closure_():
                lbfgs.zero_grad()
                x0_prediction = self.__sample_differentiable(cond, shape,
                      x_T=init_noise, unconditional_guidance_scale=uc_scale, 
                      unconditional_conditioning= self.uc)
                
                loss = loss_weights['latents'] * loss_fn(x0_prediction, target_latent, reduction = 'mean')
                if loss_weights['pixels'] != 0:
                    loss += loss_weights['pixels'] * utils.pixel_space_loss(self.ddim_sampler.model, x0_prediction, target_img, loss_fn)
                loss.backward()
                return loss.detach().item()


            x0_prediction = self.__sample_differentiable(cond, shape,
                      x_T=init_noise, unconditional_guidance_scale=uc_scale, 
                      unconditional_conditioning= self.uc)
                
            loss = loss_weights['latents'] * loss_fn(x0_prediction, target_latent, reduction = 'mean')
            if loss_weights['pixels'] != 0:
                loss += loss_weights['pixels'] * utils.pixel_space_loss(self.ddim_sampler.model, x0_prediction, target_img, loss_fn)
            
            if i % self.config.noise_optimization.log_every == 0:
                progress['loss'].append(loss.item())
                progress['noise'].append(init_noise.detach().cpu())
            
            pbar.set_postfix({'loss': loss.cpu().item()})

            if loss.item() < self.config.sufficient_loss:
                print(f'Ending computation with {loss.item()} done {i} steps.')
                break

            lbfgs.zero_grad()
            loss.backward()
            lbfgs.step(closure_)

        outputs = {
            'estimated_input_noise':  init_noise.detach(), 
            'estimated_conditioning': cond , 
            'initial_noise': init_noise_init,  
            'target_image_latent': target_latent, 
            'path2img': image, 
            'config_dict': self.config, 
            'reconstruction': x0_prediction.detach(), 
            'progress': progress, 
            'guidance_scale': uc_scale , 
        }

        return outputs


# taken from stable diffusion
    def add_noise(self, x0, noise, timestep_indices, ddim_use_original_steps=False):
        device= x0.device

        alphas_cumprod = self.ddim_sampler.model.ddim_alphas if ddim_use_original_steps else self.ddim_sampler.ddim_alphas
        sqrt_one_minus_alphas = self.ddim_sampler.model.ddim_sqrt_one_minus_alphas if ddim_use_original_steps else self.ddim_sampler.ddim_sqrt_one_minus_alphas
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        sqrt_one_minus_alphas = sqrt_one_minus_alphas.to(device)


        timestep_indices = timestep_indices.to(device)
        noise = noise.to(device)

        sqrt_at = torch.index_select(sqrt_alphas_cumprod, 0, timestep_indices).view(-1, 1, 1, 1).to(device)
        sqrt_one_minus_at = torch.index_select(sqrt_one_minus_alphas, 0, timestep_indices).view(-1, 1, 1, 1).to(device)

        noisy_samples = sqrt_at * x0.expand_as(noise) + sqrt_one_minus_at * noise
        return noisy_samples


    def ___prepare_batch_for_im(self, image):
        target_img = utils.load_pil(image)
        target_img = target_img.resize((self.config.shape[-2] * self.config.f, self.config.shape[-1] * self.config.f))
        # # create batch
        hflipper = T.RandomHorizontalFlip(p=1)
        resize_cropper = T.RandomResizedCrop(size=(512, 512), scale = (0.85, 0.99),ratio=(1,1))
        resized_crops = [resize_cropper(target_img) for _ in range(6)]
        transformed_imgs = [target_img, hflipper(target_img), *resized_crops]

        target_img = utils.pil2torch_batch(transformed_imgs)
        target_latent = utils.img2latent(self.ddim_sampler.model, target_img)
        return target_img, target_latent

    def __load_tokenizer_and_text_model(self, init_caption, tokenizer = None):
        version = 'openai/clip-vit-large-patch14'   
        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(version)
            self.tokenizer = tokenizer
        
        if init_caption is None:
            return  tokenizer, None
        batch_encoding = tokenizer(init_caption, truncation=True, max_length=77, return_length=True,
                                                return_overflowing_tokens=False, padding="max_length", return_tensors="pt")


        embeddings = self.ddim_sampler.model.cond_stage_model.transformer.get_input_embeddings().weight.data[batch_encoding['input_ids'][0]]
        text_tokens = embeddings.clone()
        text_tokens.requires_grad = True
            
        return tokenizer, text_tokens
            
    def __tokens2conditioning(self, tokens):
        conditioning = self.ddim_sampler.model.cond_stage_model.transformer(inputs_embeds = tokens.unsqueeze(0))['last_hidden_state']
        return conditioning



    def perform_cond_inversion_individual_timesteps(self, image_path, cond_init , optimize_tokens = True):
        self.config['optimize_tokens'] = optimize_tokens
        with torch.no_grad():
            _, target_latent = self.___prepare_batch_for_im(image_path)
            timesteps = torch.tensor(self.ddim_sampler.ddim_timesteps)

        if optimize_tokens:
            tokenizer, text_tokens = self.__load_tokenizer_and_text_model('', tokenizer = self.tokenizer)
            if cond_init is not None:
                text_tokens = cond_init.squeeze(0)
            
        prompt_repre = text_tokens.detach().clone()

        grad_mask = torch.zeros_like(prompt_repre)
        grad_mask[:self.config.conditioning_optimization.N_tokens,:] = 1.
        grad_mask = grad_mask.to(self.ddim_sampler.model.device)
        fetch_cond_init = lambda x: self.ddim_sampler.model.cond_stage_model.transformer(inputs_embeds = x.unsqueeze(0))['last_hidden_state']
        prompt_repre.requires_grad = True

        uc_scale = self.config.conditioning_optimization.uncond_guidance_scale

        adam = torch.optim.AdamW(params = [prompt_repre], lr = self.config.conditioning_optimization.lr)
        loss_fn = torch.nn.functional.mse_loss


        progress = {'loss':[], 'indices':[]}
        progress['cond'] = []
    
        timestep_indices = torch.randperm(8).view(-1).long()
        print(f'Selected timesteps: {timestep_indices}')


        pbar = tqdm(range(self.config.conditioning_optimization.opt_iters))
        for i in pbar:

            noise_ = torch.randn_like(target_latent)

            if not self.config.conditioning_optimization.fixed_timesteps:
                timestep_indices = torch.randint(low=0, high=self.config.ddim_steps, size=(self.config.conditioning_optimization.batch_size,1) ).view(-1)

            noisy_samples = self.add_noise(target_latent, noise_, timestep_indices, ddim_use_original_steps=False)

            steps_in = torch.index_select(timesteps, 0, timestep_indices).to(self.config.device)
            cond_init = fetch_cond_init(prompt_repre)

            noise_prediction = self.ddim_sampler.model.apply_model(noisy_samples, steps_in, cond_init.expand(self.config.conditioning_optimization.batch_size, -1 , -1))

            loss = loss_fn(noise_prediction, noise_, reduction = 'none').mean((1,2,3)).mean()

            
            if i % self.config.conditioning_optimization.log_every == 0:
                progress['indices'].append(timestep_indices)
                progress['loss'].append(loss.item())
                progress['cond'].append(prompt_repre.detach().cpu())

            
            pbar.set_postfix({'loss': loss.cpu().item(), 'indices':timestep_indices})

            if loss.item() < self.config.sufficient_loss:
                print(f'Ending computation with {loss.item()} done {i} steps.')
                break

            adam.zero_grad()
            loss.backward()
            prompt_repre.grad *= grad_mask 
            adam.step()

        outputs = {
            'estimated_conditioning':  prompt_repre.detach(), 
            'target_image_latent': target_latent, 
            'config_dict': self.config, 
            'optimize_tokens': optimize_tokens,
            'progress': progress, 
            'guidance_scale': uc_scale , 
        }

        return outputs


