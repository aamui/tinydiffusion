from synthetic_images import evaluate_saved_model, generate_synthetic_dataset, visualize_n_samples
from model_vae import VAE



def training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=512, use_wandb=True, unet_type='small'):
    X_train, y_train = generate_synthetic_dataset(num_train_samples)
    X_test, y_test = generate_synthetic_dataset(num_test_samples)

    visualize_n_samples(X_test, y_test, n=15)

    model = VAE()
    model.train(X_train, y_train, X_test, y_test, num_epochs=num_epochs, use_wandb=use_wandb, device=device, batch_size=batch_size)
    
    generated_images = model.generate_dataset(num_samples=500, device=device)
    visualize_n_samples(generated_images, n=5)

    return model

if __name__ == "__main__":
    training_pipeline(num_train_samples=50000, num_test_samples=10000, num_epochs=3, device='mps', batch_size=256, use_wandb=True, unet_type='medium')

    evaluate_saved_model(checkpoint_path='checkpoints/unet_medium_epoch_3.pth', test_size=10000, device='mps', number_of_steps=25, num_visualize=15)
