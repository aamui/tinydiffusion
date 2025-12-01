from synthetic_images import generate_synthetic_dataset, visualize_n_samples


def training_pipeline(model_class, num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=512, use_wandb=True, **kwargs):
    X_train, y_train = generate_synthetic_dataset(num_train_samples)
    X_test, y_test = generate_synthetic_dataset(num_test_samples)

    visualize_n_samples(X_test, y_test, n=15, file_name="test_samples.pdf")

    model = model_class(**kwargs)
    model.train_function(X_train, y_train, X_test, y_test, num_epochs=num_epochs, use_wandb=use_wandb, device=device, batch_size=batch_size)

    generated_images = model.generate_dataset(num_samples=500, device=device)
    visualize_n_samples(generated_images, n=5, file_name="generated_samples.pdf")

    return model
