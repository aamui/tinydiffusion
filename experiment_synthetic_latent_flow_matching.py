import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from model_latent_flow_matching import LatentFlowMatching
from synthetic_images import generate_synthetic_dataset, visualize_n_samples

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoch = 100
    X_train, y_train = generate_synthetic_dataset(500000, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_train.pt", normalize=True )
    X_test, y_test = generate_synthetic_dataset(100000, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_test.pt", normalize=True)

    model = LatentFlowMatching()
    model.train_autoencoder( X_train, X_test, num_epochs=100, device=device, batch_size=256,
                            encoder_name="synthetic_ae_encoder.pth",
                            decoder_name="synthetic_ae_decoder.pth")

    # # Train Latent Flow Matching
    model.train_function( X_train, y_train, X_test, y_test, num_epochs=num_epoch, device=device, batch_size=512, model_name="synthetic_lfm")
    # Sample
    model = LatentFlowMatching(load_from_path=f'checkpoints/synthetic_lfm_{num_epoch}.pth',
                                encoder_ckpt="checkpoints/synthetic_ae_encoder.pth",
                                decoder_ckpt="checkpoints/synthetic_ae_decoder.pth"
    )

    generated = model.generate_dataset(num_samples=10, number_of_steps=100, device=device)
    visualize_n_samples(generated, n=5, output_binarization=True, file_name="synthetic_lfm_generated_samples.pdf")