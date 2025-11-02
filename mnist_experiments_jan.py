import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

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


def visualize_n_samples(X_train, y_train=None, n=5):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        image = X_train[i]
        label = y_train[i] if y_train is not None else "Unknown"
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
    def __init__(self, in_ch=1, base_ch=32, time_emb_dim=64, load_from_path=None):
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

        # Load checkpoint if path is provided
        if load_from_path is not None:
            self.load_state_dict(torch.load(load_from_path, map_location='cpu'))
            print(f"Loaded model from {load_from_path}")

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


def train_model(model, X_train, y_train, X_test, y_test, num_epochs=1, use_wandb=True, device='cpu', batch_size=32, dimensionality_training=2):
    model.to(device)
    if use_wandb:
        wandb.init(project="mnist-diffusion", name="unet-small-mse-loss")
    train_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device)), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-5)

    loss_function = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        print(f"Running epoch {epoch+1}/{num_epochs}")
        model.train()
        losses = []
        for X_batch, y_batch in tqdm(train_data_loader):
            optimizer.zero_grad()

            time = torch.rand(X_batch.shape[0]).reshape((-1, 1, 1) if dimensionality_training == 2 else (-1, 1)).to(device)

            pure_noise_images = torch.randn(X_batch.shape).to(device)
            interpolated_images = time * X_batch + (1 - time) * pure_noise_images

            # Flow matching: predict the velocity (data - noise)
            velocity_target = X_batch - pure_noise_images

            predicted_velocity = model(interpolated_images, time)
            loss = loss_function(predicted_velocity, velocity_target)
            losses.append(loss.item())
            if use_wandb:
                wandb.log({"train_loss": loss.item()})
            loss.backward()

            optimizer.step()

        avg_train_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")

        model.eval()
        with torch.no_grad():
            losses = []
            for next_test_batch in test_data_loader:
                X_test_batch, y_test_batch = next_test_batch
                time_test = torch.rand(X_test_batch.shape[0]).reshape((-1, 1, 1) if dimensionality_training == 2 else (-1, 1)).to(device)
                pure_noise_test_images = torch.randn(X_test_batch.shape).to(device)
                interpolated_test_images = time_test * X_test_batch + (1 - time_test) * pure_noise_test_images

                # Flow matching: predict the velocity
                velocity_target_test = X_test_batch - pure_noise_test_images
                predicted_velocity_test = model(interpolated_test_images, time_test)
                test_loss = loss_function(predicted_velocity_test, velocity_target_test)
                losses.append(test_loss.item())
            avg_test_loss = sum(losses) / len(losses)
            print(f"After Epoch {epoch+1}, Test Loss: {avg_test_loss}")


        if use_wandb:
            wandb.log({"avg_train_loss_epoch": avg_train_loss, "avg_test_loss_epoch": avg_test_loss})

        # Save model checkpoint
        torch.save(model.state_dict(), f"checkpoints/unet_small_epoch_{epoch+1}.pth")


    if use_wandb:
        wandb.finish()

    model.to('cpu')


def generate_with_model(model, num_samples=5, number_of_steps=100, device='cpu', start_noise=None, dimensionality_generation=2):
    if start_noise is None:
        start_noise = torch.randn(num_samples, 28, 28)
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Start from pure noise at t=0
        generated_images = start_noise.to(device)
        dt = 1.0 / number_of_steps

        # Integrate from t=0 to t=1 (noise to data)
        for step in range(number_of_steps):
            time = torch.full((num_samples, 1, 1) if dimensionality_generation == 2 else (num_samples, 1), step / number_of_steps).to(device)
            # Predict velocity at current position
            velocity = model(generated_images, time)
            # Euler integration: move along the flow
            generated_images = generated_images + velocity * dt
    model.to('cpu')
    return generated_images.to('cpu')


def example_load_and_generate(checkpoint_path, num_samples=5, number_of_steps=100, device='cpu'):
    model = UNetSmall(load_from_path=checkpoint_path)

    generated_images = generate_with_model(
        model, 
        num_samples=num_samples, 
        number_of_steps=number_of_steps, 
        device=device
    )

    return generated_images


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist_datasets()
    visualize_n_samples(X_train, y_train, n=5)
    model = UNetSmall()
    train_model(model, X_train, y_train, X_test, y_test, num_epochs=50, use_wandb=True, device='mps', batch_size=512)
    generated_images = generate_with_model(model)
    visualize_n_samples(generated_images, n=5)

    # Example: Load from checkpoint and generate
    # generated_images = example_load_and_generate('checkpoints/unet_small_epoch_20.pth', num_samples=5, device='mps')
    # visualize_n_samples(generated_images, n=min(5, len(generated_images)))
