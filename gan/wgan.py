import torch
import torch.nn as nn
import os
from tqdm import tqdm

class generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.network(z)


class critic(nn.Module):  # Renamed from discriminator
    def __init__(self, channels=1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),  # Changed from BatchNorm2d
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            # Removed Sigmoid - output is now unbounded
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)


class wgan_gp(nn.Module):
    def __init__(self, latent_dim=100, channels=1, model_device='cpu', load_from_path=None):
        super().__init__()
        self.device = model_device
        self.latent_dim = latent_dim
        self.G = generator(latent_dim=latent_dim, channels=channels)
        self.C = critic(channels=channels)  # Renamed from D
        
        if load_from_path is not None:
            state = torch.load(load_from_path, map_location=model_device)
            self.load_state_dict(state)
            print(f"Loaded model from {load_from_path}")
        self.to(model_device)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def gradient_penalty(self, real_imgs, fake_imgs):
        batch_size = real_imgs.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_imgs.device)
        interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
        interpolated.requires_grad_(True)
        
        # Critic scores for interpolated images
        scores = self.C(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Flatten and compute norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty: push norm toward 1
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

    def train_step(self, real_imgs, opt_G, opt_C, n_critic=5, lambda_gp=10):
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # ====== Train Critic (multiple steps) ======
        for _ in range(n_critic):
            opt_C.zero_grad()

            # Real scores
            real_scores = self.C(real_imgs)

            # Fake scores
            z = torch.randn(batch_size, self.latent_dim, device=device)
            fake_imgs = self.G(z).detach()
            fake_scores = self.C(fake_imgs)

            # Gradient penalty
            gp = self.gradient_penalty(real_imgs, fake_imgs)

            # Critic loss: maximize real - fake (minimize fake - real)
            c_loss = fake_scores.mean() - real_scores.mean() + lambda_gp * gp
            c_loss.backward()
            opt_C.step()

        # ====== Train Generator ======
        opt_G.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=device)
        fake_imgs = self.G(z)
        fake_scores = self.C(fake_imgs)

        # Generator loss: maximize fake scores (minimize negative)
        g_loss = -fake_scores.mean()
        g_loss.backward()
        opt_G.step()

        return {'c_loss': c_loss.item(), 'g_loss': g_loss.item()}

    def train_epoch(self, dataloader, opt_G, opt_C, n_critic=5, lambda_gp=10):
        self.train()
        total_g_loss = 0
        total_c_loss = 0

        for batch, _ in dataloader:
            batch = batch.to(self.device)
            losses = self.train_step(batch, opt_G, opt_C, n_critic, lambda_gp)
            total_g_loss += losses['g_loss']
            total_c_loss += losses['c_loss']
        
        n = len(dataloader)
        return {'c_loss': total_c_loss / n, 'g_loss': total_g_loss / n}

    def train_function(self, X_train, y_train, X_test, y_test, use_wandb = False, device = None,n_critic=5, lambda_gp=10,
                       num_epochs=1, batch_size=32, checkpoint_dir='checkpoints'):
        train_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train.to(self.device), y_train.to(self.device)),
            batch_size=batch_size, shuffle=True
        )

        # Note: WGAN-GP paper recommends NOT using momentum (betas=(0, 0.9))
        opt_G = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_C = torch.optim.Adam(self.C.parameters(), lr=1e-4, betas=(0.0, 0.9))

        for epoch in tqdm(range(num_epochs)):
            losses = self.train_epoch(train_data_loader, opt_G, opt_C, n_critic, lambda_gp)
            print(f"Epoch {epoch+1}/{num_epochs} | C: {losses['c_loss']:.4f} | G: {losses['g_loss']:.4f}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), f"{checkpoint_dir}/wgan_gp_epoch_{num_epochs}.pt")

    @torch.no_grad()
    def sample(self, num_samples):
        self.G.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.G(z)

    @torch.no_grad()
    def generate_dataset(self, num_samples, device="cpu"):
        self.G.eval()
        self.G.to(device)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.G(z)
        samples = samples * 0.5 + 0.5
        samples = samples.clamp(0, 1)
        self.G.to('cpu')
        return samples.cpu()