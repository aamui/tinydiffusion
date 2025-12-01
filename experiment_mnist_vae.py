from experiment_mnist_flow_matching import load_mnist_datasets, visualize_n_samples
from model_vae import VAE


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()
    visualize_n_samples(X_train, y_train, n=5, file_name="train_samples_mnist_vae.pdf")
    model = VAE()
    model.train_function(X_train, y_train, X_test, y_test, num_epochs=5, use_wandb=True, device='mps', batch_size=512)

    # Example: Load from checkpoint and generate
    model = VAE(load_from_path='checkpoints/vae_epoch_5.pth')
    generated_images = model.generate_dataset(num_samples=15, device='mps')
    visualize_n_samples(generated_images, n=min(15, len(generated_images)), file_name="generated_samples_mnist_vae.pdf")
