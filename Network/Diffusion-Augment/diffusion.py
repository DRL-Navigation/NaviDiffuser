import torch, numpy, math
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from typing import Dict


# beta schedule
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def sigmoid_beta_schedule(timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas)/(betas.max()-betas.min())*(0.02-betas.min())/10
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ClassifyFreeDDIM:
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule='linear',
        ddim_discr_method='linear',
        ddim_timesteps=50,
        ddim_eta=0.0,
        guide_dropout = 0.2,
        guide_w = 2
    ):
        self.timesteps = timesteps
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.ddim_timesteps = ddim_timesteps
        if ddim_discr_method == 'linear':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = numpy.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (numpy.linspace(0, numpy.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise ValueError(f'unknown ddim discretization method called {ddim_discr_method}')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        self.ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        self.ddim_timestep_prev_seq = numpy.append(numpy.array([0]), self.ddim_timestep_seq[:-1])
        self.ddim_eta = ddim_eta

        self.guide_dropout = guide_dropout
        self.guide_w = guide_w
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        index = t.squeeze(1).to(device=a.device, dtype=torch.long)
        out = a.gather(0, index).to(device=t.device, dtype=torch.float32)
        out = out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def _q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # use ddim to sample
    @torch.no_grad()
    def sample(self, x : Dict[str, torch.Tensor], c : Dict[str, torch.Tensor]):
        self.eval()
        batch_size, device = x['action'].shape[0], x['action'].device
        
        for i in reversed(range(0, self.ddim_timesteps)):
            t = torch.full((batch_size, 1), self.ddim_timestep_seq[i], device=device, dtype=torch.float32)
            prev_t = torch.full((batch_size, 1), self.ddim_timestep_prev_seq[i], device=device, dtype=torch.float32)

            pred_noise_c_dict = self.model.sample(x, t, c, torch.ones(batch_size).bool().to(device=device))
            pred_noise_none_dict = self.model.sample(x, t, c, torch.zeros(batch_size).bool().to(device=device))
            
            x.pop("time")
            for key in x.keys():
                if key == "condition" or key == "condition_mask": continue
                # 1. get current and previous alpha_cumprod
                alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x[key].shape)
                alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, x[key].shape)
    
                # 2. predict noise using model
                pred_noise = (1+self.guide_w)*pred_noise_c_dict[key] - self.guide_w*pred_noise_none_dict[key]
            
                # 3. get the predicted x_0
                pred_x0 = (x[key] - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            
                # 4. compute variance: "sigma_t(η)" -> see formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = self.ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
                # 5. compute "direction pointing to x_t" of formula (12)
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
                # 6. compute x_{t-1} of formula (12)
                x[key] = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x[key])
            
        return x
    
    # compute train losses
    def learn(self, x : Dict[str, torch.Tensor], c : Dict[str, torch.Tensor]):
        self.train()
        noise_dict = deepcopy(x)
        noise_dict.pop('condition')
        batch_size, device = x['action'].shape[0], x['action'].device
        t = torch.randint(0, self.timesteps, (batch_size, 1)).to(device=device, dtype=torch.float32)
        c_mask = (torch.rand(batch_size) > self.guide_dropout).to(device)
        for key in noise_dict.keys():
            noise_dict[key] = torch.randn_like(x[key])
            x[key] = self._q_sample(x[key], t, noise_dict[key])
        predicted_noise_dict = self.model(x, t, c, c_mask)
        loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
        for key in noise_dict.keys():
            loss = loss + F.mse_loss(noise_dict[key], predicted_noise_dict[key])
        return loss
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def load(self, path):
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path))
    
    def save(self, name):
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)