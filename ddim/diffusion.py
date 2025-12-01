import torch
import os
from .noise_schedule import extract
from tqdm import tqdm
from .networks import DiffusionCNN
from .noise_schedule import NoiseScheduler




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
		for batch, labels in dataloader:
			# assumes batch is just images
			loss = self.train_step(batch, optimizer)
			total_loss += loss
		batch_avg_loss = total_loss / len(dataloader)
		return batch_avg_loss 

	def eval_epoch(self, dataloader):
		self.model.eval()
		total_loss = 0
		loss_fn = torch.nn.MSELoss()
		for batch, labels in dataloader:
			batch = batch.to(self.device)
			bs = batch.shape[0]
			t = torch.randint(0, self.noise_scheduler.timesteps, (bs,), device=self.device)
			xt, noise = self.forward_diffusion(batch, t)
			pred_noise = self.model(xt, t)
			loss = loss_fn(pred_noise, noise)
			total_loss += loss.item()
		return total_loss / len(dataloader)

	def train(self, X_train, y_train, X_test, y_test, num_epochs=1, use_wandb=False, batch_size=32, checkpoint_dir='checkpoints', noise_schedule = 'linear'):

		train_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(self.device), y_train.to(self.device)), 
			batch_size=batch_size, shuffle=True)

		test_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.to(self.device), y_test.to(self.device)), 
			batch_size=batch_size, shuffle=False)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-4)

		for epoch in tqdm(range(num_epochs)):
			avg_loss = self.train_epoch(train_data_loader, optimizer)
			print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f}")

			with torch.no_grad():
				test_avg_loss = self.eval_epoch(test_data_loader)
				print(f'After Epoch {epoch + 1}/{num_epochs} |  Test Loss: {test_avg_loss:.4f}')
		os.makedirs(checkpoint_dir, exist_ok=True)
		torch.save(self.model.state_dict(), f"{checkpoint_dir}/ddim_epoch_{epoch+1}.pt")


	@torch.no_grad()
	def ddim_sample(self, bs, shape, inference_steps):
		# shape should be same as img we want. ie Mnist - bs, 1, 28, 28
		self.model.eval()
		channel, h, w = shape
		img = (bs, channel, h, w)
		step_size = self.noise_scheduler.timesteps // inference_steps

		sample_ratio = self.noise_scheduler.timesteps // inference_steps
		sample_timesteps = torch.arange(0, self.noise_scheduler.timesteps, sample_ratio, device = self.device) # for example [0, 50, 100, 150, ...1000]
		sample_timesteps = sample_timesteps.flip(0) # then flip it and [1000, 950, 900, 850, ...]

		x = torch.randn(img, device = self.device)
		for i, t in enumerate(sample_timesteps):
			prev_t = t - step_size
			t_batched = torch.full((bs,), t, device = self.device)
			prev_t_batched = torch.full((bs,), prev_t, device=self.device)
			pred_noise = self.model(x, t_batched)
			sqrt_one_minus_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t_batched, x.shape)
			sqrt_inv_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_inv_alphas_cumprod, t_batched, x.shape)
			pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) * sqrt_inv_alphas_cumprod_t

			if prev_t >= 0:
				alphas_cumprod_prev_t = extract(self.noise_scheduler.alphas_cumprod, prev_t_batched, x.shape)
			else:
				alphas_cumprod_prev_t = torch.ones_like(sqrt_one_minus_alphas_cumprod_t)
			x = torch.sqrt(alphas_cumprod_prev_t) * pred_x0 + (torch.sqrt(1 - alphas_cumprod_prev_t) * pred_noise)
		return x









