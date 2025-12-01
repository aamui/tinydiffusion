import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from networks import DiffusionCNN
from noise_schedule import NoiseScheduler
from diffusion import diffusion
import argparse
import os


def main():
	parser = argparse.ArgumentParser(description = 'DDIM Sampling')
	parser.add_argument("--num_samples", type = int, default = 10, help = 'Number of samples')
	parser.add_argument("--inference_steps", type = int, default = 50, help = "Number of inference steps")
	parser.add_argument("--noise_schedule", type = str, default="linear", help = "linear OR cosine")
	parser.add_argument("--model_path", type = str, default = "./results/ddim_mnist_model.pt", help = "path to saved model")
	parser.add_argument("--save_file", type = str, default = "samples.png", help= "save file name")
	args = parser.parse_args()

	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"
	print(f"Using device: {device}")

	model = DiffusionCNN(image_channels = 1, time_emb_dim = 128, base_channels = 64)
	model.to(device)
	state = torch.load(args.model_path, map_location=device)
	model.load_state_dict(state)

	noise_scheduler = NoiseScheduler(timesteps = 1000, schedule = args.noise_schedule)
	diff = diffusion(model, noise_scheduler, device = device)

	save_dir = 'results'
	sample_dir = os.path.join(save_dir,"samples")
	os.makedirs(sample_dir, exist_ok=True)

	print('Sampling...')
	samples = diff.ddim_sample(bs = args.num_samples, shape = (1, 28, 28), inference_steps = args.inference_steps)

	samples = (samples + 1) / 2 # un transform
	samples = torch.clamp(samples, 0, 1)
	grid = make_grid(samples, nrow = 8)
	samples_path = os.path.join(sample_dir, args.save_file)
	save_image(grid, samples_path)

	print(f'Samples saved to {samples_path}')



if __name__ == "__main__":
	main()




