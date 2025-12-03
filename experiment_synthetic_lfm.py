from synthetic_images import evaluate_saved_model
from training_pipeline import training_pipeline_lfm
from model_latent_flow_matching import Encoder, Decoder, UNetSmall, LatentFlowMatching
import time
import torch

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    training_pipeline_lfm(LatentFlowMatching, num_train_samples=500000, num_test_samples=100000, num_epochs=150, device=device, batch_size=256)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(LatentFlowMatching, checkpoint_path='checkpoints/synthetic_lfm_epoch_150.pth', test_size=10000, device=device, num_visualize=15)
