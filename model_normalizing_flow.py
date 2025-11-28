import math
import torch
import torch.nn as nn
from model_abstract import Model
from tqdm import tqdm

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=512, mask=None):
        super().__init__()
        self.dim = dim

        if mask is None:
            mask = torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim - dim // 2)
            ])
        self.register_buffer("mask", mask)

        # network that outputs [s, t]
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim)
        )

    def forward(self, x, reverse=False):
        """
        x: [B, D]
        returns: y, log_det (per-sample)
        """
        m = self.mask  
        x_masked = x * m  

        # s, t conditioned on masked part
        st = self.net(x_masked)         
        s, t = st.chunk(2, dim=1)        
        s = torch.tanh(s)

        if not reverse:
            # y1 = x1
            # y2 = x2 * exp(s2) + t2
            y = x_masked + (1 - m) * (x * torch.exp(s) + t)
            log_det = ((1 - m) * s).sum(dim=1)  
        else:
            # x2 = (y2 - t2) * exp(-s2)
            y = x_masked + (1 - m) * ((x - t) * torch.exp(-s))
            log_det = -((1 - m) * s).sum(dim=1)

        return y, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=512, num_flows=4):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows

        masks = []
        for i in range(num_flows):
            if i % 2 == 0:
                mask = torch.cat([
                    torch.ones(dim // 2),
                    torch.zeros(dim - dim // 2)
                ])
            else:
                mask = torch.cat([
                    torch.zeros(dim // 2),
                    torch.ones(dim - dim // 2)
                ])
            masks.append(mask)

        self.couplings = nn.ModuleList([
            AffineCoupling(dim, hidden_dim=hidden_dim, mask=masks[i])
            for i in range(num_flows)
        ])

    def forward(self, x, reverse=False):
        """
        x: [B, D]
        forward (reverse=False)  : z -> u (to base Gaussian)
        forward (reverse=True)   : u -> z (sampling)
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        if not reverse:
            # z -> u
            for c in self.couplings:
                x, log_det = c(x, reverse=False)
                log_det_total += log_det
        else:
            # u -> z (inverse order)
            for c in reversed(self.couplings):
                x, log_det = c(x, reverse=True)
                log_det_total += log_det

        return x, log_det_total

class LatentNormalizingFlow(Model):
    def __init__(self, latent_ch=4, latent_hw=7,
                 hidden_dim=512, num_flows=4, load_from_path=None):
        super().__init__()
        self.latent_ch = latent_ch
        self.latent_hw = latent_hw
        self.latent_dim = latent_ch * latent_hw * latent_hw

        self.flow = RealNVP(
            dim=self.latent_dim,
            hidden_dim=hidden_dim,
            num_flows=num_flows
        )

        if load_from_path is not None:
            state = torch.load(load_from_path, map_location="cpu")
            self.flow.load_state_dict(state)

    def train_autoencoder(self, encoder, decoder,
                          X_train, X_test,
                          num_epochs=5, device='cpu', batch_size=256):
        encoder.to(device); decoder.to(device)
        ds_train = torch.utils.data.TensorDataset(X_train.to(device))
        ds_test  = torch.utils.data.TensorDataset(X_test.to(device))
        tl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        vl = torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=2e-3, weight_decay=1e-5
        )
        mse = nn.MSELoss()

        for epoch in range(num_epochs):
            encoder.train(); decoder.train()
            tr_losses = []
            for (xb,) in tl:
                opt.zero_grad()
                x = xb.unsqueeze(1)  # [B,1,28,28]
                z = encoder(x)
                x_hat = decoder(z)
                loss = mse(x_hat, x)
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())
            avg_tr = sum(tr_losses) / len(tr_losses)

            # val
            encoder.eval(); decoder.eval()
            with torch.no_grad():
                v_losses = []
                for (xb,) in vl:
                    x = xb.unsqueeze(1)
                    x_hat = decoder(encoder(x))
                    v_losses.append(mse(x_hat, x).item())
            avg_v = sum(v_losses) / len(v_losses)
            print(f"[AE] Epoch {epoch+1}/{num_epochs} - train {avg_tr:.4f} | val {avg_v:.4f}")
        encoder.to('cpu'); decoder.to('cpu')


    def train_nf(self, encoder,
                 X_train, X_test,
                 num_epochs=10, device='cpu', batch_size=512):
        """
        encoder: trained encoder (frozen during NF training)
        X_train/X_test: [N, 28, 28] tensors
        """
        # freeze encoder
        encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        self.flow.to(device).train()

        ds_train = torch.utils.data.TensorDataset(X_train.to(device))
        ds_test  = torch.utils.data.TensorDataset(X_test.to(device))
        tl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        vl = torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(self.flow.parameters(), lr=1e-4, weight_decay=2e-5)

        D = self.latent_dim
        const_gauss = 0.5 * D * math.log(2 * math.pi)

        for epoch in range(num_epochs):
            print(f"[NF] Running epoch {epoch+1}/{num_epochs}")
            self.flow.train()
            tr_losses = []

            for (xb,) in tqdm(tl):
                opt.zero_grad()
                x = xb.unsqueeze(1).to(device)  

                with torch.no_grad():
                    z = encoder(x)  
                z = z.view(z.shape[0], -1)     

                # z -> u
                u, log_det = self.flow(z, reverse=False)  

                # log p(u) for standard Gaussian
                log_p_u = -0.5 * (u ** 2).sum(dim=1) - const_gauss  
                log_p_z = log_p_u + log_det                         
                loss = -log_p_z.mean()  # NLL

                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            avg_tr = sum(tr_losses) / len(tr_losses)

            # validation (just NLL)
            self.flow.eval()
            with torch.no_grad():
                v_losses = []
                for (xb,) in vl:
                    x = xb.unsqueeze(1).to(device)
                    z = encoder(x)
                    z = z.view(z.shape[0], -1)
                    u, log_det = self.flow(z, reverse=False)
                    log_p_u = -0.5 * (u ** 2).sum(dim=1) - const_gauss
                    log_p_z = log_p_u + log_det
                    loss = -log_p_z.mean()
                    v_losses.append(loss.item())

            avg_v = sum(v_losses) / len(v_losses)
            print(f"[NF] Epoch {epoch+1} - train {avg_tr:.4f} | val {avg_v:.4f}")

        self.flow.to('cpu')


    def generate_with_model(self, decoder,
                            num_samples=5, device='cpu'):
 
        self.flow.to(device).eval()
        decoder.to(device).eval()

        D = self.latent_dim
        with torch.no_grad():
            # sample from base Gaussian
            u = torch.randn(num_samples, D).to(device)
            # invert flow: u -> z
            z_flat, _ = self.flow(u, reverse=True)
            z = z_flat.view(num_samples, self.latent_ch, self.latent_hw, self.latent_hw)
            x_hat = decoder(z).squeeze(1)  

        self.flow.to('cpu')
        decoder.to('cpu')
        return x_hat.cpu()
