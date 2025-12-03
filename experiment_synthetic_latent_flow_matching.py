import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from model_latent_flow_matching import Encoder, Decoder, UNetSmall, LatentFlowMatching
from synthetic_images import generate_synthetic_dataset, visualize_n_samples

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = generate_synthetic_dataset(500000, use_saved=False, save_path="./syntetic_data_normal_train.pt", normalize=True )
    X_test, y_test = generate_synthetic_dataset(100000, use_saved=False, save_path="./syntetic_data_normal_test.pt", normalize=True)

    # Pretrain AE
    encoder = Encoder(in_ch=1, latent_ch=4)
    decoder = Decoder(out_ch=1, latent_ch=4)

    model = LatentFlowMatching()
    model.train_autoencoder( X_train, X_test, num_epochs=150, device=device, batch_size=256)

    # Train Latent Flow Matching
    num_epochs = 150
    model.train_function(X_train, y_train, X_test, y_test, num_epochs=num_epochs, device=device, batch_size=512)

    # Sample
    model = LatentFlowMatching(load_from_path=f'checkpoints/latent_flow_epoch_{num_epochs}.pth')
    generated = model.generate_dataset(num_samples=10, number_of_steps=100, device=device)
    visualize_n_samples(generated, n=5, file_name="lfm_generated_samples.pdf")

