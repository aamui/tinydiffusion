import torch
import torch.nn as nn


class generator(nn.Module):
	def __init__(self, latent_dim = 100, channels = 1):
		super().__init__()

		self.network = nn.Sequential(
			nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias = False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),

			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),

			nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
			nn.Tanh()
			)

	def forward(self, z):
		# input: batch, latent_dim -> batch, latent_dim, 1, 1
		# output: batch, 1, 28, 28 (for mnist)
		z = z.view(z.size(0), z.size(1), 1, 1)
		return self.network(z)


class discriminator(nn.Module):
	def __init__(self, channels = 1):
		super().__init__()

		self.network = nn.Sequential(
			nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(128, 1, 7, 1, 0, bias=False),
			nn.Sigmoid()
			)


	def forward(self, x):
	# input: batch, 1, 28, 28 (for mnist)
	# output: batch, 1 
		return self.network(x).view(-1, 1)


class dcgan(nn.Module):
	def __init__(self, latent_dim = 100, channels = 1, device = 'mps'):
		super().__init__()
		self.device = device
		self.latent_dim = latent_dim
		self.G = generator(latent_dim = latent_dim, channels = channels)
		self.D = discriminator(channels = channels)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
			nn.init.normal_(m.weight, 0.0, 0.02)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.zeros_(m.bias)


	def train_step(self, real_imgs, opt_G, opt_D, dsteps = 1):
		batch_size = real_imgs.size(0)
		device = real_imgs.device

		real_labels = torch.ones(batch_size, 1, device=device)
		fake_labels = torch.zeros(batch_size, 1, device=device)
		criterion = nn.BCELoss()

		# training discriminator
		for i in range(dsteps):
			opt_D.zero_grad()

			d_loss_real = criterion(self.D(real_imgs), real_labels)

			z = torch.randn(batch_size, self.latent_dim, device=device)
			fake_imgs = self.G(z).detach()
			d_loss_fake = criterion(self.D(fake_imgs), fake_labels)

			d_loss = d_loss_real + d_loss_fake
			d_loss.backward()
			opt_D.step()

		# training generator
		opt_G.zero_grad()

		z = torch.randn(batch_size, self.latent_dim, device=device)
		fake_imgs = self.G(z)
		g_loss = criterion(self.D(fake_imgs), real_labels)

		g_loss.backward()
		opt_G.step()

		return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}

	def train_epoch(self, dataloader, opt_G, opt_D, dsteps = 1):
		total_g_loss = 0
		total_d_loss = 0

		for batch, _ in dataloader:
			batch = batch.to(self.device)
			losses = self.train_step(batch, opt_G, opt_D, dsteps)
			total_g_loss += losses['g_loss']
			total_d_loss += losses['d_loss']
		n = len(dataloader)
		return {'d_loss': total_d_loss / n, 'g_loss': total_g_loss / n}



	@torch.no_grad()
	def sample(self, num_samples, device='mps'):
		z = torch.randn(num_samples, self.latent_dim, device=device)
		return self.G(z)




