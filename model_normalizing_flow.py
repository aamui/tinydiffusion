import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model_abstract import Model


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=512, mask=None):
        super().__init__()
        self.dim = dim

        # mask: [D]
        if mask is None:
            mask = torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim - dim // 2)
            ])
        self.register_buffer("mask", mask)

        # network that outputs [s, t] in R^D x R^D
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
        returns: y, log_det (per-sample) with shape [B]
        """
        # m: [D] -> broadcast to [B, D]
        m = self.mask  # [D]
        # 브로드캐스트는 자동이지만, 명시적으로 [1, D]로 만들어두고 싶다면:
        # m = self.mask.view(1, -1)

        x_masked = x * m  # [B, D]

        # s, t conditioned on masked part
        st = self.net(x_masked)        # [B, 2D]
        s, t = st.chunk(2, dim=1)      # [B, D], [B, D]
        s = torch.tanh(s)              # 안정성 위해 scale 제한

        if not reverse:
            # forward: x -> y
            # y1 = x1
            # y2 = x2 * exp(s2) + t2
            y = x_masked + (1 - m) * (x * torch.exp(s) + t)
            log_det = ((1 - m) * s).sum(dim=1)  # [B]
        else:
            # inverse: y -> x
            # x2 = (y2 - t2) * exp(-s2)
            y = x_masked + (1 - m) * ((x - t) * torch.exp(-s))
            log_det = -((1 - m) * s).sum(dim=1)  # [B]

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
        forward (reverse=False)  : x -> u (to base Gaussian)
        forward (reverse=True)   : u -> x (sampling)
        returns: y, log_det_total
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        if not reverse:
            # x -> u
            for c in self.couplings:
                x, log_det = c(x, reverse=False)
                log_det_total += log_det
        else:
            # u -> x (inverse order)
            for c in reversed(self.couplings):
                x, log_det = c(x, reverse=True)
                log_det_total += log_det

        return x, log_det_total


class NormalizingFlow(Model):
    def __init__(self, dim, hidden_dim=1024, num_flows=8, load_from_path=None):
        super().__init__()
        self.dim = dim

        self.flow = RealNVP(
            dim=dim,
            hidden_dim=hidden_dim,
            num_flows=num_flows
        )

        if load_from_path is not None:
            state = torch.load(load_from_path, map_location="cpu")
            self.flow.load_state_dict(state)

    def forward(self, x, reverse=False):
        """
        x: [B, D]  (항상 벡터 형태로 사용)
        """
        return self.flow(x, reverse=reverse)

    def train_function(self, X_train, X_test,
                 num_epochs=10, device='cpu', batch_size=512, checkpoint_dir='checkpoints'):

        self.flow.to(device).train()

        ds_train = torch.utils.data.TensorDataset(X_train.to(device))
        ds_test  = torch.utils.data.TensorDataset(X_test.to(device))
        tl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        vl = torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(self.flow.parameters(), lr=1e-4, weight_decay=2e-5)

        D = self.dim
        const_gauss = 0.5 * D * math.log(2 * math.pi)

        for epoch in range(num_epochs):
            print(f"[NF] Running epoch {epoch+1}/{num_epochs}")
            self.flow.train()
            tr_losses = []

            for (xb,) in tqdm(tl):
                opt.zero_grad()
                # xb: [B, D] 라고 가정 (이미 flatten 되어 있음)
                x = xb.to(device)  # [B, D]

                # x -> u
                u, log_det = self.flow(x, reverse=False)  # u: [B, D], log_det: [B]

                # log p(u) for standard Gaussian N(0, I)
                log_p_u = -0.5 * (u ** 2).sum(dim=1) - const_gauss  # [B]
                log_p_x = log_p_u + log_det                          # [B]
                loss = -log_p_x.mean()  # NLL

                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            avg_tr = sum(tr_losses) / len(tr_losses)

            # validation (just NLL)
            self.flow.eval()
            with torch.no_grad():
                v_losses = []
                for (xb,) in vl:
                    x = xb.to(device)
                    u, log_det = self.flow(x, reverse=False)
                    log_p_u = -0.5 * (u ** 2).sum(dim=1) - const_gauss
                    log_p_x = log_p_u + log_det
                    loss = -log_p_x.mean()
                    v_losses.append(loss.item())

            avg_v = sum(v_losses) / len(v_losses)
            print(f"[NF] Epoch {epoch+1} - train {avg_tr:.4f} | val {avg_v:.4f}")

        torch.save(self.model.state_dict(), f"{checkpoint_dir}/normalizing_flow_epoch_{epoch+1}.pth")
        self.flow.to('cpu')

    def generate(self, num_samples=5, device='cpu'):
        self.flow.to(device).eval()

        D = self.dim
        with torch.no_grad():
            # sample from base Gaussian
            u = torch.randn(num_samples, D).to(device)  # [B, D]
            # invert flow: u -> x
            x, _ = self.flow(u, reverse=True)           # [B, D]

        self.flow.to('cpu')
        return x.cpu()

    def generate_dataset(self, num_samples, device='cpu', 
                         max_images_per_batch=2048):
        generated_images = []

        for batch_start in tqdm(range(0, num_samples, max_images_per_batch)):
            batch_end = min(batch_start + max_images_per_batch, num_samples)
            batch_size = batch_end - batch_start
            batch_images = self.generate(
                num_samples=batch_size, 
                device=device
            )
            generated_images.append(batch_images)
        
        return torch.cat(generated_images, dim=0)
