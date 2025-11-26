from synthetic_images import evaluate_saved_model, generate_synthetic_dataset, visualize_n_samples
from unet import UNetSmall, UNetMedium
from flow_matching import train_model, generate_with_model



def training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=512, use_wandb=True, unet_type='small'):
    X_train, y_train = generate_synthetic_dataset(num_train_samples)
    X_test, y_test = generate_synthetic_dataset(num_test_samples)

    # visualize_n_samples(X_test, y_test, n=15)

    model = UNetSmall() if unet_type == 'small' else UNetMedium()
    train_model(model, X_train, y_train, X_test, y_test, num_epochs=num_epochs, use_wandb=use_wandb, device=device, batch_size=batch_size, unet_type=unet_type)

    # generated_images = generate_with_model(model)
    # visualize_n_samples(generated_images, n=5)

    return model

if __name__ == "__main__":
    training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=256, use_wandb=True, unet_type='medium')

    evaluate_saved_model('checkpoints/unet_medium_epoch_50.pth', test_size=100000, device='mps', number_of_steps=25)
