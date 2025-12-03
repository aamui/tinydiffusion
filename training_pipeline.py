from synthetic_images import generate_synthetic_dataset, visualize_n_samples
import torch

def training_pipeline(model_class, num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=512, use_wandb=True, **kwargs):
    X_train, y_train = generate_synthetic_dataset(num_train_samples)
    X_test, y_test = generate_synthetic_dataset(num_test_samples)

    visualize_n_samples(X_test, y_test, n=15, file_name="test_samples.pdf")

    model = model_class(**kwargs)
    model.train_function(X_train, y_train, X_test, y_test, num_epochs=num_epochs, use_wandb=use_wandb, device=device, batch_size=batch_size)

    generated_images = model.generate_dataset(num_samples=500, device=device)
    visualize_n_samples(generated_images, n=5, file_name="generated_samples.pdf")

    return model

def training_pipeline_lfm(model_class, num_train_samples=500000, num_test_samples=100000, num_epochs=150, device='mps', batch_size=512, **kwargs):
    X_train, y_train = generate_synthetic_dataset(num_train_samples, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_train.pt", normalize=True )
    X_test, y_test = generate_synthetic_dataset(num_test_samples, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_test.pt", normalize=True)

    model = model_class(**kwargs)
    model.train_autoencoder( X_train, X_test, num_epochs=num_epochs, device=device, batch_size=256,
                            encoder_name="synthetic_ae_encoder.pth",
                            decoder_name="synthetic_ae_decoder.pth")

    model.train_function( X_train, y_train, X_test, y_test, num_epochs=num_epochs, device=device, batch_size=512, model_name="synthetic_lfm")

    model = model_class(load_from_path=f'checkpoints/synthetic_lfm_epoch_{num_epochs}.pth',
                                encoder_ckpt="checkpoints/synthetic_ae_encoder.pth",
                                decoder_ckpt="checkpoints/synthetic_ae_decoder.pth", **kwargs)

    generated_images = model.generate_dataset(num_samples=500, number_of_steps=150, device=device)
    visualize_n_samples(generated_images, n=5, output_binarization=True, file_name="synthetic_lfm_generated_samples.pdf")

    return model


def training_pipeline_nf(model_class, num_train_samples=500000, num_test_samples=100000, num_epochs=150, device='mps', batch_size=512, **kwargs):

    X_train, y_train = generate_synthetic_dataset(num_train_samples, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_train.pt", normalize=True )
    X_test, y_test = generate_synthetic_dataset(num_test_samples, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_test.pt", normalize=True)

    num_classes = int(y_train.max().item() + 1)
    X_train = X_train.view(X_train.size(0), -1)
    X_test  = X_test.view(X_test.size(0), -1)
    dim = X_train.size(1)

    model = model_class(dim=dim, num_classes=num_classes, **kwargs)
    model.train_function(X_train, y_train,X_test, y_test,num_epochs=num_epochs,device=device,model_name=f"eval_syn_nf")

    # model.train_function(X_train, y_train, X_test, y_test, num_epochs=num_epochs, device=device, batch_size=batch_size, model_name="syn_nf")
    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    class_probs = class_counts / class_counts.sum()

    model = model_class(dim=dim,num_classes=num_classes,load_from_path=f"./checkpoints/eval_syn_nf_epoch_{num_epochs}.pth",  **kwargs)
    

    num_samples = 500
    samples = model.generate_dataset(num_samples=num_samples, device=device, class_probs=class_probs)
    samples_img = samples.view(num_samples, 1, 28, 28)

    visualize_n_samples(samples_img, n=5, output_binarization=True, file_name=f"eval_syn_nf_generated_samples.pdf")
