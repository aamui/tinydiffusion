import time
from synthetic_images import evaluate_saved_model
from train_pipeline_am import training_pipeline_am
import torch
from ddim.diffusion import diffusion
from ddim.networks import DiffusionCNN
from ddim.noise_schedule import NoiseScheduler
import os
import torchvision
from torchvision import transforms
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--noise_schedule", type = str, default="linear", help = "linear OR cosine")
    parser.add_argument("--file_name", type = str, default = "ddim_samples.pdf", help = "sample file name")
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
    timesteps = 1000
    noise_scheduler = NoiseScheduler(timesteps = timesteps, schedule = args.noise_schedule)

    training_pipeline_am(diffusion, num_train_samples=50000, num_test_samples=10000, num_epochs=args.epochs, device=device, batch_size=args.batch_size, use_wandb=False,file_name = args.file_name, model = model, noise_scheduler = noise_scheduler, model_device = device)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(diffusion, checkpoint_path=f'checkpoints/ddim_epoch_{args.epochs}.pt', test_size=1000, device=device, num_visualize=15)




if __name__ == "__main__":
    main()
