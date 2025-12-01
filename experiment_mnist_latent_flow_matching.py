import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from model_latent_flow_matching import Encoder, Decoder, UNetSmall, LatentFlowMatching

def load_mnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    X_train = torch.stack([train_dataset[i][0].reshape(28, 28) for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0].reshape(28, 28) for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    return (X_train, y_train), (X_test, y_test)

def visualize_n_samples(X_train, y_train=None, n=5, title="Samples"):
    X_train = X_train.detach().cpu()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        image = X_train[i]
        label = y_train[i] if y_train is not None else "Unknown"
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()

    # Pretrain AE
    encoder = Encoder(in_ch=1, latent_ch=4)
    decoder = Decoder(out_ch=1, latent_ch=4)

    model = LatentFlowMatching(model_type='medium')
    model.train_autoencoder(encoder, decoder, X_train, X_test, num_epochs=5, device=device, batch_size=256)

    # Train Latent Flow Matching
    model.train_lfm(encoder, decoder, X_train, y_train, X_test, y_test, num_epochs=150, device=device, batch_size=512)

    # Sample
    model = LatentFlowMatching(model_type='medium', load_from_path='checkpoints/latent_flow_epoch_150.pth', dimensionality=2)
    generated = model.generate_with_model(decoder, num_samples=10, number_of_steps=100, device=device, latent_ch=4, latent_hw=7)

    visualize_n_samples(generated, n=5, title="LFM Generated")