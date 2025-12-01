from experiment_mnist_flow_matching import load_mnist_datasets, visualize_n_samples
import torch
from ddim.diffusion import diffusion
from ddim.networks import DiffusionCNN
from ddim.noise_schedule import NoiseScheduler


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = load_mnist_datasets()
	if X_train.dim() == 3:
		X_train = X_train.unsqueeze(1)
		X_test = X_test.unsqueeze(1)
		
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"
	print(f"Using device: {device}")

	model = DiffusionCNN(image_channels = 1, time_emb_dim = 128, base_channels = 64)
	model.to(device)
	timesteps = 1000
	noise_scheduler = NoiseScheduler(timesteps = timesteps, schedule = 'linear')
	diff = diffusion(model, noise_scheduler, device = device)
	diff.train(X_train, y_train, X_test, y_test, num_epochs = 1, use_wandb=False, batch_size=32, checkpoint_dir='checkpoints')
