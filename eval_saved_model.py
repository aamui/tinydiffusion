import torch
import argparse
from synthetic_images import evaluate_saved_model_threshold
from ddim.diffusion import diffusion

def main():
	parser = argparse.ArgumentParser(description="Train a Diffusion Model")
	parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
	parser.add_argument("--samples", type=int, default=1000, help="num samples")
	args = parser.parse_args()

	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"
	else:
		device = "cpu"
	print(f"Using device: {device}")


	evaluate_saved_model_threshold(diffusion, checkpoint_path=f'checkpoints/ddim_epoch_100.pt', test_size=args.samples, device=device, num_visualize=15, threshold = args.threshold)


if __name__ == "__main__":
	main()