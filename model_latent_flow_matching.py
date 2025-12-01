import torch
import torch.nn as nn
from model_abstract import Model
from tqdm import tqdm


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

# Simple Conv AutoEncoder for MNIST: x(1x28x28) <-> z(Cx7x7)
class Encoder(nn.Module):
    def __init__(self, in_ch=1, latent_ch=4):
        super().__init__()
        # 28 -> 14
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1),
            nn.SiLU()
        )
        # 14 -> 7
        self.e2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU()
        )
        self.to_latent = nn.Conv2d(32, latent_ch, 1)

    def forward(self, x):  # x: [B,1,28,28]
        h = self.e1(x)     # [B,16,14,14]
        h = self.e2(h)     # [B,32,7,7]
        z = self.to_latent(h)  # [B,C,7,7]
        return z

class Decoder(nn.Module):
    def __init__(self, out_ch=1, latent_ch=4):
        super().__init__()
        # 7 -> 14
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 32, 4, stride=2, padding=1),
            nn.SiLU()
        )
        # 14 -> 28
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.SiLU()
        )
        self.to_img = nn.Conv2d(16, out_ch, 1)

    def forward(self, z):  # z: [B,C,7,7]
        h = self.d1(z)     # [B,32,14,14]
        h = self.d2(h)     # [B,16,28,28]
        x_hat = self.to_img(h)  # [B,1,28,28]
        return x_hat

class UNetSmall(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, time_emb_dim=128, load_from_path=None):
        super().__init__()

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, base_ch)  
        )

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)        
        self.enc2 = ConvBlock(base_ch, base_ch * 2)   
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        # Decoder (skip-concat)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 2 + base_ch, base_ch)

        # Output in latent space
        self.out = nn.Conv2d(base_ch, in_ch, 1)

        # Load checkpoint if path is provided
        if load_from_path is not None:
            self.load_state_dict(torch.load(load_from_path, map_location='cpu'))
            print(f"Loaded model from {load_from_path}")

    def forward(self, z_t, t):
        """
        z_t: [B, C, 7, 7] latent input
        t  : [B, 1]      time in [0,1]
        """
        t_feat = self.time_mlp(t)       
        t_feat = t_feat[:, :, None, None]  

        e1 = self.enc1(z_t)
        e1 = e1 + t_feat               
        e2 = self.enc2(e1)
        b  = self.bottleneck(e2)

        d1 = self.dec1(torch.cat([b, e2], dim=1))
        d2 = self.dec2(torch.cat([d1, e1], dim=1))
        out = self.out(d2)              
        return out
    


class LatentFlowMatching(Model):
    def __init__(self, model_type="small", load_from_path=None):
        self.model_type = model_type
        self.model = UNetSmall(load_from_path=load_from_path)


    def train_autoencoder(self, encoder, decoder, X_train, X_test, num_epochs=5, device='cpu', batch_size=256):
        encoder.to(device); decoder.to(device)
        ds_train = torch.utils.data.TensorDataset(X_train.to(device))
        ds_test  = torch.utils.data.TensorDataset(X_test.to(device))
        tl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        vl = torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-3, weight_decay=1e-5)
        mse = nn.MSELoss()

        for epoch in range(num_epochs):
            encoder.train(); decoder.train()
            tr_losses = []
            for (xb,) in tl:
                opt.zero_grad()
                x = xb.unsqueeze(1)  
                z = encoder(x)
                x_hat = decoder(z)
                loss = mse(x_hat, x)
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())
            avg_tr = sum(tr_losses)/len(tr_losses)

            # val
            encoder.eval(); decoder.eval()
            with torch.no_grad():
                v_losses = []
                for (xb,) in vl:
                    x = xb.unsqueeze(1)
                    x_hat = decoder(encoder(x))
                    v_losses.append(mse(x_hat, x).item())
            avg_v = sum(v_losses)/len(v_losses)
            print(f"[AE] Epoch {epoch+1}/{num_epochs} - train {avg_tr:.4f} | val {avg_v:.4f}")
        encoder.to('cpu'); decoder.to('cpu')

    def train_lfm(self, encoder, decoder, X_train, y_train, X_test, y_test,
                num_epochs=10, device='cpu', batch_size=512, checkpoint_dir='checkpoints'):
        # freeze AE
        encoder.to(device).eval()
        decoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False

        self.model.to(device).train()

        train_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device)), batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=2e-5)
        loss_function = nn.MSELoss()

        for epoch in range(num_epochs):
            print(f"[LFM] Running epoch {epoch+1}/{num_epochs}")
            self.model.train()
            tr_losses = []

            for (X_batch, _) in tqdm(train_data_loader):
                opt.zero_grad()

                X_batch = X_batch.unsqueeze(1).to(device)  # [B,1,28,28]
                with torch.no_grad():
                    latent = encoder(X_batch)  
                pure_noise_latent = torch.randn_like(latent)  # [B,C,7,7]

                time = torch.rand(X_batch.shape[0], 1).to(device)  # [B,1]
                t_b = time[:, :, None, None]  # [B,1,1,1] for broadcast

                interpolated_latent = (1 - t_b) * pure_noise_latent + t_b * latent 
                velocity_target = latent - pure_noise_latent
                predicted_velocity = self.model(interpolated_latent, time)

                loss = loss_function(predicted_velocity, velocity_target)
                loss.backward()
                opt.step()

                tr_losses.append(loss.item())
            avg_tr = sum(tr_losses) / len(tr_losses)

    
            self.model.eval()
            with torch.no_grad():
                v_losses = []
                for (X_test_batch, _) in test_data_loader: 
                    X_test_batch = X_test_batch.unsqueeze(1).to(device)
                    latent = encoder(X_test_batch)
                    pure_noise_latent = torch.randn_like(latent)

                    time_test = torch.rand(X_test_batch.shape[0], 1).to(device) 
                    t_b = time_test[:, :, None, None]

                    interpolated_latent = (1 - t_b) * pure_noise_latent + t_b * latent
                    target_v = latent - pure_noise_latent
                    pred_v = self.model(interpolated_latent, time_test)
                    v_losses.append(loss_function(pred_v, target_v).item())

            avg_v = sum(v_losses) / len(v_losses)
            print(f"[LFM] Epoch {epoch+1} - train {avg_tr:.4f} | val {avg_v:.4f}")

        torch.save(self.model.state_dict(), f"{checkpoint_dir}/latent_flow_epoch_{epoch+1}.pth")
        self.model.to('cpu')

    def generate_with_model(self, decoder, num_samples=5, number_of_steps=100,
                            device='cpu', latent_ch=4, latent_hw=7):
        self.model.to(device).eval()
        decoder.to(device).eval()
        with torch.no_grad():
            latent = torch.randn(num_samples, latent_ch, latent_hw, latent_hw).to(device)
            dt = 1.0 / number_of_steps
            for step in range(number_of_steps):
                time = torch.full((num_samples, 1), step / number_of_steps).to(device)
                velocity = self.model(latent, time)
                latent = latent + velocity * dt
            generated_images = decoder(latent).squeeze(1)
        self.model.to('cpu')
        decoder.to('cpu')
        return generated_images.cpu()






