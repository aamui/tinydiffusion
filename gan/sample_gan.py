import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

from gan import dcgan


def main():
	parser = argparse.ArgumentParser(description="Train a DC-GAN Model")
	parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
	parser.add_argument("--model_path", type = str, default = "./results/test.pt", help = "path to saved model")
	parser.add_argument("--save_file", type = str, default = "samples.png", help= "save file name")
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
	state = torch.load(args.model_path, map_location=device)

	gan = dcgan(latent_dim=100, channels=1).to(device)
	gan.load_state_dict(state)
	gan.eval()

	print("Sampling...")
	samples = gan.sample(args.num_samples, device = device)

	samples = (samples + 1) / 2 # un transform
	samples = torch.clamp(samples, 0, 1)
	grid = make_grid(samples, nrow = 8)
	samples_path = os.path.join(sample_dir, args.save_file)
	save_image(grid, samples_path)




if __name__ == "__main__":
	main()