import torch
import os
from synthetic_images import sample_from_model
from ddim.diffusion import diffusion

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    sample_from_model(diffusion, checkpoint_path = 'checkpoints/ddim_epoch_20.pt', samples = 50, device = device)


if __name__ == "__main__":
    main()
