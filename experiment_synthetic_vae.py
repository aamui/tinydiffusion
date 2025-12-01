import time
from synthetic_images import evaluate_saved_model
from model_vae import VAE
from training_pipeline import training_pipeline


if __name__ == "__main__":
    training_pipeline(VAE, num_train_samples=50000, num_test_samples=10000, num_epochs=3, device='mps', batch_size=256, use_wandb=False)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(VAE, checkpoint_path='checkpoints/vae_epoch_3.pth', test_size=10000, device='mps', num_visualize=15)
