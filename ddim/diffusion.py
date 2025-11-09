import torch
from noise_schedule import extract


class diffusion:
	def __init__(self, model, noise_scheduler, device = 'cpu'):
		self.model = model.to(device)
		self.noise_scheduler = noise_scheduler.to(device)
		self.device = device
	def forward_diffusion(self, x0, t, noise = None):
		if noise is None:
			noise = torch.randn_like(x0)

		sqrt_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x0.shape)
		sqrt_one_minus_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x0.shape)

		xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

		return xt, noise

	def train_step(self, x0, optimizer):
		x0 = x0.to(self.device)
		loss_fn = torch.nn.MSELoss()
		optimizer.zero_grad()

		bs = x0.shape[0]
		t = torch.randint(0, self.noise_scheduler.timesteps, (bs,), device = self.device)

		xt, noise = self.forward_diffusion(x0, t)
		pred_noise = self.model(xt, t)

		loss = loss_fn(pred_noise, noise)

		loss.backward()
		optimizer.step()

		return loss.item()

	def train_epoch(self, dataloader, optimizer):
		self.model.train()
		total_loss = 0
		for batch in dataloader:
			# assumes batch is just images
			loss = self.train_step(batch, optimizer)
			total_loss += loss
		batch_avg_loss = total_loss / len(dataloader)
		return batch_avg_loss 


	@torch.no_grad()
	def ddim_sample(self, bs, shape, inference_steps):
		# shape should be same as img we want. ie Mnist - bs, 1, 28, 28
		self.model.eval()
		channels, h, w = shape
		img = (bs, channels, h, w)

		sample_ratio = self.noise_scheduler.timesteps // inference_steps
		sample_timesteps = torch.arange(0, self.noise_scheduler.timesteps, sample_ratio, device = self.device) # for example [0, 50, 100, 150, ...1000]
		sample_timesteps = sample_timesteps.flip(0) # then flip it and [1000, 950, 900, 850, ...]

		x = torch.randn(img, device = self.device)
		for i, t in enumerate(sample_timesteps):
			t_batched = torch.full((bs,), t, device = self.device)
			pred_noise = self.model(x, t_batched)
			sqrt_one_minus_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t_batched, x.shape)
			sqrt_inv_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_inv_alphas_cumprod, t_batched, x.shape)
			pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) * sqrt_inv_alphas_cumprod_t

			alphas_cumprod_prev_t = extract(self.noise_scheduler.alphas_cumprod_prev, t_batched, x.shape)
			x = torch.sqrt(alphas_cumprod_prev_t) * pred_x0 + (torch.sqrt(1 - alphas_cumprod_prev_t) * pred_noise)
		return x








