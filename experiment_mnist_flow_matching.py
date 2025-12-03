import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model_flow_matching import FlowMatching
import os


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


def visualize_n_samples(X_train, y_train=None, n=5, output_binarization=False, file_name="please_specify_filename.pdf", threshold = 0.5):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    file_name = os.path.join("plots", file_name)

    if output_binarization:
        X_train = (X_train > threshold).float()

    n_rows = max(1, (n + 4) // 5)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        image = X_train[i]
        
        if image.ndim == 1 and image.numel() == 28 * 28:
            image = image.view(28, 28)
        elif image.ndim == 2 and image.shape == (28 * 28, 1):
            image = image.view(28, 28)
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        
        label = y_train[i] if y_train is not None else "Unknown"
        axes[i // 5, i % 5].imshow(image.squeeze(), cmap='gray')
        axes[i // 5, i % 5].set_title(f'Label: {label}')
        axes[i // 5, i % 5].axis('off')
    plt.savefig(file_name)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()
    visualize_n_samples(X_train, y_train, n=5, file_name="train_samples.pdf")
    model = FlowMatching(model_type='medium', dimensionality=2)
    model.train_function(X_train, y_train, X_test, y_test, num_epochs=3, use_wandb=True, device='mps', batch_size=512)
    generated_images = model.generate(num_samples=500, device='mps')
    visualize_n_samples(generated_images, n=5, file_name="generated_samples.pdf")

    # Example: Load from checkpoint and generate
    model = FlowMatching(model_type='medium', load_from_path='checkpoints/unet_medium_epoch_3.pth', dimensionality=2)
    generated_images = model.generate(num_samples=15, device='mps')
    visualize_n_samples(generated_images, n=min(15, len(generated_images)), file_name="generated_samples_from_checkpoint.pdf")
