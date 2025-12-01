# from experiment_mnist_flow_matching import load_mnist_datasets, visualize_n_samples
import torch
from ddim.diffusion import diffusion
from ddim.networks import DiffusionCNN
from ddim.noise_schedule import NoiseScheduler
import os
from torchvision.utils import save_image, make_grid
import torchvision
from torchvision import transforms
import argparse



def load_mnist_datasets():
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.50,), (0.50,))])
	train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

	print(f"Number of training samples: {len(train_dataset)}")
	print(f"Number of test samples: {len(test_dataset)}")

	X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
	y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
	X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
	y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

	return (X_train, y_train), (X_test, y_test)

def main():
	parser = argparse.ArgumentParser(description="Train a Diffusion Model")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--noise_schedule", type = str, default="linear", help = "linear OR cosine")
	args = parser.parse_args()

	(X_train, y_train), (X_test, y_test) = load_mnist_datasets()
	
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"
	print(f"Using device: {device}")
	print(f'Total training epochs: {args.epochs}')
	print(f'Training batch size: {args.batch_size}')
	print(f'Noise scheudle: {args.noise_schedule}')

	model = DiffusionCNN(image_channels = 1, time_emb_dim = 128, base_channels = 64)
	model.to(device)
	timesteps = 1000
	noise_scheduler = NoiseScheduler(timesteps = timesteps, schedule = args.noise_schedule)
	diff = diffusion(model, noise_scheduler, device = device)
	# epochs = 20
	diff.train(X_train, y_train, X_test, y_test, num_epochs = args.epochs, use_wandb=False, batch_size=args.batch_size, checkpoint_dir='checkpoints')
	print('Training done!')

	print('Sampling...')
	samples = diff.ddim_sample(bs = 10, shape = (1, 28, 28), inference_steps = 50)

	samples = (samples + 1) / 2 # un transform
	samples = torch.clamp(samples, 0, 1)
	grid = make_grid(samples, nrow = 8)
	os.makedirs('samples', exist_ok=True)
	samples_path = os.path.join('samples', f'ddim_epochs{args.epochs}.png')
	save_image(grid, samples_path)

	print(f'Samples saved to {samples_path}')


if __name__ == "__main__":
	main()

