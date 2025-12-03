import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from model_latent_flow_matching import LatentFlowMatching
from synthetic_images import  visualize_n_samples

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

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch = 5
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()

    model = LatentFlowMatching()
    model.train_autoencoder( X_train, X_test, num_epochs=5, device=device, batch_size=256,
                            encoder_name="mnist_ae_encoder.pth",
                            decoder_name="mnist_ae_decoder.pth")

    # Train Latent Flow Matching
    model.train_function( X_train, y_train, X_test, y_test, num_epochs=num_epoch, device=device, batch_size=512)

    # Sample
    model = LatentFlowMatching(load_from_path=f"checkpoints/mnist_lfm_epoch_{num_epoch}.pth",
        encoder_ckpt="checkpoints/mnist_ae_encoder.pth",
        decoder_ckpt="checkpoints/mnist_ae_decoder.pth",
    )
    generated = model.generate_dataset( num_samples=10, number_of_steps=100, device=device)
    visualize_n_samples(generated, n=5, output_binarization=True, file_name="mnist_LFM_Generated.pdf")