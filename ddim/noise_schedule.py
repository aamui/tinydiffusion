import torch
import numpy as np

def linear_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
	# timesteps is total timesteps like T=0,...1000. thus 1000
	return torch.linspace(beta_start, beta_end, timesteps)

def cosine_schedule(timesteps, s):
	steps = timesteps + 1
	t = torch.linspace(0, timesteps, steps)
	f = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
	alphas_cumprod = f / f[0] # need this to normalize, because first index needs to be exactly 1 (clean img).
	'''
	bellow comes from:
	alpha_bar_t (cumprod) = alpha_0 * ... * alpha_t
	beta_t = 1-alpha_t
	alpha_t = alpha_bar_t / alpha_0 *...*alpha_t-1
	and beta_0 = alpha_bar_1 / alpha_bar_0 , thus the offset [1:] and [:-1] in the numerator/denominator respectively
	''' 
	betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clip(betas, 0.0001, 0.9999)


class NoiseScheduler:
	def __init__(self, timesteps = 1000, schedule = 'linear'):
		self.timesteps = timesteps
		if schedule == 'linear':
			betas = linear_schedule(self.timesteps)
		if schedule == 'cosine':
			betas = cosine_schedule(self.timesteps, s=0.008)

		self.betas = betas # 1,T vector
		self.alphas = 1. - betas # 1,T vector

		self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0) # alpha bar
		self.alphas_cumprod_prev = torch.cat((torch.ones(1), self.alphas_cumprod[:-1]))
		
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
		
		self.sqrt_inv_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)	
		self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt((1. / self.alphas_cumprod) - 1)

	def to(self, device):
		self.betas = self.betas.to(device)
		self.alphas = self.alphas.to(device)

		self.alphas_cumprod = self.alphas_cumprod.to(device)
		self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)

		self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
		self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

		self.sqrt_inv_alphas_cumprod = self.sqrt_inv_alphas_cumprod.to(device)
		self.sqrt_inv_alphas_cumprod_minus_one = self.sqrt_inv_alphas_cumprod_minus_one.to(device)
		return self



def extract(a, t, x_shape):
	bs = t.shape[0]
	out = a.gather(-1, t)
	return out.reshape(bs, *((1,) * (len(x_shape) - 1)))

		



 
