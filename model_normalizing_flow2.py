import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model_abstract import Model


class AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim=0, hidden_dim=512, mask=None):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim

        if mask is None:
            mask = torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim - dim // 2)
            ])
        self.register_buffer("mask", mask)   # [D]

        in_dim = dim + cond_dim  # x_masked concat y

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim)
        )

    def forward(self, x, y=None, reverse=False):
        """
        x: [B, D]
        y: [B, C] (one-hot or embedding), or None for unconditional
        """
        m = self.mask  # [D]
        x_masked = x * m  # [B, D]

        if y is not None:
            inp = torch.cat([x_masked, y], dim=1)  # [B, D+C]
        else:
            inp = x_masked  # [B, D]

        st = self.net(inp)          # [B, 2D]
        s, t = st.chunk(2, dim=1)   # [B, D], [B, D]
        s = torch.tanh(s)

        if not reverse:
            y_out = x_masked + (1 - m) * (x * torch.exp(s) + t)
            log_det = ((1 - m) * s).sum(dim=1)
        else:
            y_out = x_masked + (1 - m) * ((x - t) * torch.exp(-s))
            log_det = -((1 - m) * s).sum(dim=1)

        return y_out, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, cond_dim=0, hidden_dim=512, num_flows=4):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
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
            AffineCoupling(dim, cond_dim=cond_dim,
                           hidden_dim=hidden_dim, mask=masks[i])
            for i in range(num_flows)
        ])

    def forward(self, x, y=None, reverse=False):
        """
        x: [B, D]
        y: [B, C] or None
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)

        if not reverse:
            for c in self.couplings:
                x, log_det = c(x, y=y, reverse=False)
                log_det_total += log_det
        else:
            for c in reversed(self.couplings):
                x, log_det = c(x, y=y, reverse=True)
                log_det_total += log_det

        return x, log_det_total


class NormalizingFlow(Model):
    def __init__(self, dim=784, num_classes=10,
                 hidden_dim=1024, num_flows=8, load_from_path=None):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        self.flow = RealNVP(
            dim=dim,
            cond_dim=num_classes,
            hidden_dim=hidden_dim,
            num_flows=num_flows
        )

        if load_from_path is not None:
            state = torch.load(load_from_path, map_location="cpu")
            self.flow.load_state_dict(state)

    def forward(self, x, y=None, reverse=False):
        return self.flow(x, y=y, reverse=reverse)

    def _one_hot(self, y):
        return torch.nn.functional.one_hot(
            y.long(), num_classes=self.num_classes
        ).float()

    def train_function(self, X_train, y_train, X_test, y_test,
                       num_epochs=10, device='cpu', batch_size=512,
                       checkpoint_dir='checkpoints'):

        self.flow.to(device).train()

        ds_train = torch.utils.data.TensorDataset(
            X_train.to(device), y_train.to(device))
        ds_test = torch.utils.data.TensorDataset(
            X_test.to(device), y_test.to(device))
        tl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,
                                         shuffle=True)
        vl = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,
                                         shuffle=False)

        opt = torch.optim.AdamW(self.flow.parameters(),
                                lr=1e-4, weight_decay=2e-5)

        D = self.dim
        const_gauss = 0.5 * D * math.log(2 * math.pi)

        for epoch in range(num_epochs):
            print(f"[NF] Running epoch {epoch+1}/{num_epochs}")
            self.flow.train()
            tr_losses = []

            for xb, yb in tqdm(tl):
                opt.zero_grad()
                x = xb.to(device)          # [B, D]
                y = self._one_hot(yb).to(device)  # [B, C]

                u, log_det = self.flow(x, y=y, reverse=False)

                log_p_u = -0.5 * (u ** 2).sum(dim=1) - const_gauss
                log_p_x_given_y = log_p_u + log_det
                loss = -log_p_x_given_y.mean()

                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            avg_tr = sum(tr_losses) / len(tr_losses)

            self.flow.eval()
            with torch.no_grad():
                v_losses = []
                for xb, yb in vl:
                    x = xb.to(device)
                    y = self._one_hot(yb).to(device)
                    u, log_det = self.flow(x, y=y, reverse=False)
                    log_p_u = -0.5 * (u ** 2).sum(dim=1) - const_gauss
                    log_p_x_given_y = log_p_u + log_det
                    loss = -log_p_x_given_y.mean()
                    v_losses.append(loss.item())

            avg_v = sum(v_losses) / len(v_losses)
            print(f"[NF] Epoch {epoch+1} - train {avg_tr:.4f} | val {avg_v:.4f}")

        torch.save(self.flow.state_dict(),
                   f"{checkpoint_dir}/conditional_nf_epoch_{epoch+1}.pth")
        self.flow.to('cpu')

    def generate(self, num_samples=5, y=None, device='cpu'):
        """
        y: [B] labels or None.
        If None, sample labels uniformly.
        """
        self.flow.to(device).eval()

        if y is None:
            y = torch.randint(0, self.num_classes, (num_samples,))
        elif y.dim() == 0:
            y = y.unsqueeze(0).repeat(num_samples)
        else:
            assert y.size(0) == num_samples

        y_oh = self._one_hot(y).to(device)

        D = self.dim
        with torch.no_grad():
            u = torch.randn(num_samples, D).to(device)
            x, _ = self.flow(u, y=y_oh, reverse=True)

        self.flow.to('cpu')
        return x.cpu()
