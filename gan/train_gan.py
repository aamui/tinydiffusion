import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

from gan import dcgan


def main():
	parser = argparse.ArgumentParser(description="Train a DC-GAN Model")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--model_path", type=str, default="model.pt", help="Model name")
	args = parser.parse_args()



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

	gan = dcgan(latent_dim=100, channels=1).to(device)
	opt_G = torch.optim.Adam(gan.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
	opt_D = torch.optim.Adam(gan.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

	for epoch in range(args.epochs):
		losses = gan.train_epoch(train_loader, opt_G, opt_D, dsteps=1)
		print(f"Epoch {epoch+1}/{args.epochs} | D: {losses['d_loss']:.4f} | G: {losses['g_loss']:.4f}")

	model_path = os.path.join(save_dir, args.model_path)
	torch.save(gan.state_dict(), model_path)


if __name__ == "__main__":
	main()






