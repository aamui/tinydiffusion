import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        time_out = self.time_mlp(time_emb)
        h = h + time_out[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)


class DiffusionCNN(nn.Module):
    def __init__(self, image_channels=1, time_emb_dim=128, base_channels=64):
        super().__init__()
        
        self.time_embedding = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # in conv
        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        
        # down sample
        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        
        # mid
        self.mid1 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # up sample
        self.up1 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # out conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)
        
        h = self.conv_in(x)
        
        h1 = self.down1(h, t_emb)      
        h = self.pool(h1)              
        
        h2 = self.down2(h, t_emb)     
        h = self.pool(h2)               
        
        h = self.mid1(h, t_emb)        
        h = self.mid2(h, t_emb)       
        
        h = self.upsample(h)           
        h = h + h2                     
        h = self.up1(h, t_emb)        
        
        h = self.upsample(h)           
        h = h + h1                      
        h = self.up2(h, t_emb)         
        
        h = self.conv_out(h)            
        
        return h