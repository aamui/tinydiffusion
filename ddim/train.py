import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms


from networks import DiffusionCNN
from noise_schedule import NoiseScheduler
from diffusion import diffusion


def main():
	parser = argparse.ArgumentParser(description="Train a Diffusion Model")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--noise_schedule", type = str, default="linear", help = "linear OR cosine")
	# parser.add_argument("--device", type=str, default="mps", help="Device (cuda, cpu, or mps)")
	args = parser.parse_args()

	timesteps = 1000
	n_samples = 10
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"
	print(f"Using device: {device}")

	save_dir = 'results'
	sample_dir = os.path.join(save_dir, 'samples')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(sample_dir, exist_ok=True)

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

	train_dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)

	train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)

	model = DiffusionCNN(image_channels = 1, time_emb_dim = 128, base_channels = 64)
	model.to(device)
	noise_scheduler = NoiseScheduler(timesteps = timesteps, schedule = args.noise_schedule)
	diff = diffusion(model, noise_scheduler, device = device)
	optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

	for epoch in range(args.epochs):
		avg_loss = diff.train_epoch(train_loader, optimizer)
		print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")
		# could add code to log loss
	print('Sampling...')
	samples = diff.ddim_sample(bs = n_samples, shape = (1, 28, 28), inference_steps = 50)

	samples = (samples + 1) / 2 # un transform
	samples = torch.clamp(samples, 0, 1)
	grid = make_grid(samples, nrow = 8)
	samples_path = os.path.join(sample_dir, 'ddim_mnist_samples.png')
	save_image(grid, samples_path)

	model_path = os.path.join(save_dir, 'ddim_mnist_model.pt')
	torch.save(model.state_dict(), model_path)
	print(f'Model saved to {model_path}')
	print(f'Samples saved to {samples_path}')



if __name__ == "__main__":
	main()


