import time
from synthetic_images import evaluate_saved_model
from train_pipeline_am import training_pipeline_am
import torch
import argparse
from gan.gan import dcgan


def main():
    parser = argparse.ArgumentParser(description="Train a GAN")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--dsteps", type=int, default=1, help="Number of discriminator steps")
    parser.add_argument("--file_name", type = str, default = "gan_samples.pdf", help = "sample file name")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    #model = dcgan(latent_dim=100, channels=1, device = device).to(device)
    training_pipeline_am(dcgan, num_train_samples=50000, num_test_samples=10000, num_epochs=args.epochs, device=device, batch_size=args.batch_size, use_wandb=False,file_name = args.file_name, latent_dim = 100, channels = 1, model_device = device)
   
    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(dcgan, checkpoint_path=f'checkpoints/gan_epoch_{args.epochs}.pt', test_size=1000, device=device, num_visualize=15)


if __name__ == "__main__":
    main()
