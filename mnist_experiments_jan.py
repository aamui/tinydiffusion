import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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


def visualize_n_samples(X_train, y_train, n=5):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        image = X_train[i]
        label = y_train[i]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.legend()
    plt.show()



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, time_emb_dim=64):
        super().__init__()

        # time embedding (for diffusion or flow matching)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 2 + base_ch, base_ch)

        # Output
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        x = x.unsqueeze(1)  # Ensure input has channel dimension
        # Embed time and add it as conditioning
        t_emb = self.time_mlp(t.view(-1, 1))
        t_emb = t_emb[:, :, None, None]  # reshape to [batch, time_emb_dim, 1, 1]
        # Rescale to match input channel dimension for addition
        t_emb_scaled = t_emb / (t_emb.abs().max() + 1e-8) * 0.1  # scaled down

        # Encoder
        e1 = self.enc1(x)  # process image normally
        e2 = self.enc2(self.pool(e1))

        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder
        d1 = self.dec1(torch.cat([self.up(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e1], dim=1))
        out = self.out(d2)
        return out.reshape(-1, 28, 28)



if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()
    # visualize_n_samples(X_train, y_train, n=5)
    model = UNetSmall()
    # Add channel dimension: [batch, height, width] -> [batch, channels, height, width]
    output = model(X_train[:20], torch.tensor([0.5]))
    print(f"Model output shape: {output.shape}")
