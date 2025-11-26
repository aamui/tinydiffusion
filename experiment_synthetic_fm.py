from synthetic_images import evaluate_saved_model, generate_synthetic_dataset, visualize_n_samples
from unet import UNetSmall, UNetMedium
from flow_matching import FlowMatching



def training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=512, use_wandb=True, unet_type='small'):
    X_train, y_train = generate_synthetic_dataset(num_train_samples)
    X_test, y_test = generate_synthetic_dataset(num_test_samples)

    visualize_n_samples(X_test, y_test, n=15)

    model = FlowMatching(model_type=unet_type, dimensionality=2)
    model.train(X_train, y_train, X_test, y_test, num_epochs=num_epochs, use_wandb=use_wandb, device=device, batch_size=batch_size)

    generated_images = model.generate(num_samples=500, device=device, number_of_steps=25)
    visualize_n_samples(generated_images, n=5)

    return model

if __name__ == "__main__":
    training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=256, use_wandb=True, unet_type='medium')

    # trained_model = FlowMatching(model_type='medium', dimensionality=2)
    # trained_modelevaluate_saved_model('checkpoints/unet_medium_epoch_50.pth', test_size=100000, device='mps', number_of_steps=25)
