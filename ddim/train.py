import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms


from networks import DiffusionCNN
from noise_schedule import NoiseScheduler
from diffusion import diffusion


def main():
	bs = 64
	epochs = 100
	lr = 1e-4
	timesteps = 1000
	device = 'cpu'
	n_samples = 10

	save_dir = 'results'
	sample_dir = os.path.join(save_dir, 'samples')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(sample_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

	train_dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)

	train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers = 2)

	model = DiffusionCNN(image_channels = 1, time_emb_dim = 128, base_channels = 64)
	noise_scheduler = NoiseScheduler(timesteps = timesteps, schedule = 'linear')
	diff = diffusion(model, noise_scheduler, device = device)
	optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

	for epoch in range(epochs):
		avg_loss = diff.train_epoch(train_loader, optimizer)
		# could add code to log loss

	samples = diff.ddim_sample(bs = n_samples, shape = (1, 28, 28), inference_steps = 50)

	samples = (samples + 1) / 2 # un transform
	samples = torch.clamp(samples, 0, 1)
	grid = make_grid(samples, nrow = 8)
	samples_path = os.path.join(sample_dir, 'ddim_mnist_samples.png')
	save_image(grid, samples_path)

	model_path = os.path.join(save_dir, 'ddim_mnist_model.pt')
	torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
	main()


