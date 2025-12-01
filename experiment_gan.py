import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from gan.gan import dcgan



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




if __name__ == "__main__":

	(X_train, y_train), (X_test, y_test) = load_mnist_datasets()
	
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"
	print(f"Using device: {device}")
	epochs = 1
	gan = dcgan(latent_dim=100, channels=1, device = device).to(device)
	gan.train_model(X_train, y_train, X_test, y_test, num_epochs = epochs, use_wandb=False, batch_size=32, checkpoint_dir='checkpoints')	
	print('Training done!')

	print('Sampling...')
	num_samples = 10
	samples = gan.sample(num_samples, device = device)
	samples = (samples + 1) / 2 # un transform
	samples = torch.clamp(samples, 0, 1)
	grid = make_grid(samples, nrow = 8)
	os.makedirs('samples', exist_ok=True)
	samples_path = os.path.join('samples', f'gan_epochs{epochs}.png')
	save_image(grid, samples_path)

	print(f'Samples saved to {samples_path}')




