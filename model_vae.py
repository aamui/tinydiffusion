import torch
import torch.nn as nn
from model_abstract import Model
from unet import ConvBlock
import wandb
from tqdm import tqdm


class VAE(nn.Module, Model):
    def __init__(self, in_ch=1, latent_dim=16, base_ch=32, load_from_path=None):
        super().__init__()
        self.model_type = 'vae'
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool = nn.AvgPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        
        # After enc3 and pooling: (base_ch * 4) x 7 x 7 for 28x28 input
        self.flatten_dim = base_ch * 4 * 7 * 7
        
        # Latent space (mean and log variance)
        self.fully_connected_encode = nn.Linear(self.flatten_dim, self.flatten_dim)
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        
        # Decoder
        self.fully_connected_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.fc_decode = nn.Linear(self.flatten_dim, self.flatten_dim)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)
        self.dec3 = ConvBlock(base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, in_ch, 1)

        if load_from_path is not None:
            self.load_state_dict(torch.load(load_from_path, map_location='cpu'))
            print(f"Loaded VAE model from {load_from_path}")

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.enc1(x))   # 14x14
        x = self.pool(self.enc2(x))   # 7x7
        x = self.enc3(x)              # 7x7
        x = x.view(x.size(0), -1)     # Flatten
        x = self.fully_connected_encode(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fully_connected_decode(z)
        x = self.fc_decode(x)
        x = x.view(x.size(0), -1, 7, 7)  # Reshape to feature map
        x = self.up(self.dec1(x))         # 14x14
        x = self.up(self.dec2(x))         # 28x28
        x = self.dec3(x)
        x = self.out(x)
        return x.reshape(-1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples=1, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def train(self, X_train, y_train, X_test, y_test, num_epochs=1, use_wandb=True, 
              device='cpu', batch_size=32, checkpoint_dir='checkpoints', kl_weight=1e-3):
        if use_wandb:
            wandb.init(project="mnist-diffusion", name=f"unet-{self.model_type}-mse-loss")
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=2e-5)
    
        self.to(device)

        train_data_loader, test_data_loader, loss_function = self._prepare_training(X_train, y_train, X_test, y_test, checkpoint_dir, batch_size, device)

        for epoch in tqdm(range(num_epochs)):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            losses = []
            for X_batch, y_batch in tqdm(train_data_loader):
                optimizer.zero_grad()

                reconstructed, mu, logvar = self(X_batch)
                recon_loss = loss_function(reconstructed, X_batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_weight * kl_loss

                losses.append(loss.item())
                if use_wandb:
                    wandb.log({"train_loss": loss.item(), "recon_loss": recon_loss.item(), "kl_loss": kl_loss.item()})
                loss.backward()

                optimizer.step()

            avg_train_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")

            with torch.no_grad():
                losses = []
                for next_test_batch in test_data_loader:
                    X_test_batch, y_test_batch = next_test_batch
                    reconstructed_test, mu_test, logvar_test = self(X_test_batch)
                    recon_loss_test = loss_function(reconstructed_test, X_test_batch)
                    kl_loss_test = -0.5 * torch.mean(1 + logvar_test - mu_test.pow(2) - logvar_test.exp())
                    test_loss = recon_loss_test + kl_weight * kl_loss_test
                    losses.append(test_loss.item())
                avg_test_loss = sum(losses) / len(losses)
                print(f"After Epoch {epoch+1}, Test Loss: {avg_test_loss}")

            if use_wandb:
                wandb.log({"avg_train_loss_epoch": avg_train_loss, "avg_test_loss_epoch": avg_test_loss})

            # Save model checkpoint
            torch.save(self.state_dict(), f"{checkpoint_dir}/vae_epoch_{epoch+1}.pth")

        if use_wandb:
            wandb.finish()

        self.to('cpu')
        
    def generate_dataset(self, num_samples, device='cpu'):
        generated_images = []
        
        self.to(device)
        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                img = self.sample(num_samples=1, device=device)
                generated_images.append(img.cpu())
        self.to('cpu')
        return torch.cat(generated_images, dim=0)