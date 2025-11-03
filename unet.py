import torch
import torch.nn as nn


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


class UNetMedium(nn.Module):
    """
    Medium-sized UNet with 3 encoder/decoder levels and more channels.
    Architecture:
    - Base channels: 64 (vs 32 in Small)
    - Encoder levels: 3 (vs 2 in Small)
    - Time embedding dim: 128 (vs 64 in Small)
    - Parameters: ~4-5x more than UNetSmall
    """
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128, load_from_path=None):
        super().__init__()

        # time embedding (for diffusion or flow matching)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder (3 levels)
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # Decoder (3 levels)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec3 = ConvBlock(base_ch * 2 + base_ch, base_ch)

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

        # Encoder (3 levels)
        e1 = self.enc1(x)  # 28x28 -> 28x28 (base_ch)
        e2 = self.enc2(self.pool(e1))  # 28x28 -> 14x14 (base_ch*2)
        e3 = self.enc3(self.pool(e2))  # 14x14 -> 7x7 (base_ch*4)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # 7x7 -> 3x3 (base_ch*8)

        # Decoder (3 levels)
        d1 = self.dec1(torch.cat([self.up(b), e3], dim=1))  # 3x3 -> 7x7
        d2 = self.dec2(torch.cat([self.up(d1), e2], dim=1))  # 7x7 -> 14x14
        d3 = self.dec3(torch.cat([self.up(d2), e1], dim=1))  # 14x14 -> 28x28
        out = self.out(d3)
        return out.reshape(-1, 28, 28)
