from synthetic_images import evaluate_saved_model
from training_pipeline import training_pipeline_nf
from model_normalizing_flow2 import NormalizingFlow
import time
import torch

if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"

if __name__ == "__main__":
    print(f"Using device: {device}")
    training_pipeline_nf(NormalizingFlow, num_epochs=150, device=device, batch_size=256)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(NormalizingFlow, checkpoint_path='checkpoints/normalizing_flow_epoch_150.pth', test_size=10000, device=device, num_visualize=15)
