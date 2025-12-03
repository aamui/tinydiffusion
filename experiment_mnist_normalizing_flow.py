import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model_normalizing_flow import NormalizingFlow

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

alpha = 1e-6

def preprocess_x(x):
    # x: [B, 1, 28, 28], 0..1
    # 1) dequant (간단히 uniform noise 추가)
    x = x + torch.rand_like(x) / 256.0      # if 0..255 였으면 /256, 여기선 그냥 작은 noise여도 됨
    x = torch.clamp(x, 0.0, 1.0)

    # 2) logit transform: x ∈ (0,1) -> y ∈ R
    x = alpha + (1 - 2*alpha) * x           # avoid 0/1
    y = torch.log(x) - torch.log(1 - x)     # logit

    return y

def postprocess_y(y):
    # y: [B, D] in R (logit space)
    # inverse logit
    x = torch.sigmoid(y)   # ∈ (0,1)
    return x

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()

    # # Flatten to [B, D]
    # X_train = X_train.view(X_train.size(0), -1)
    # X_test  = X_test.view(X_test.size(0), -1)

    # main 쪽
    (X_train, _), (X_test, _) = load_mnist_datasets()
    X_train = preprocess_x(X_train).view(X_train.size(0), -1)
    X_test  = preprocess_x(X_test).view(X_test.size(0), -1)


    nf_model = NormalizingFlow(dim=X_train.size(1))
    nf_model.train_nf(X_train, X_test, num_epochs=150, device=device)

    # Sample
    # samples = nf_model.generate_with_model(num_samples=16, device=device) 
    # samples = samples.view(-1, 1, 28, 28) 

    samples = nf_model.generate_with_model(num_samples=16, device=device)  # [16,D]
    samples = postprocess_y(samples)
    samples = samples.view(-1, 1, 28, 28)
    
    visualize_n_samples(samples, n=5, title="NF Generated")
