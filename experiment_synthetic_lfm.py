from synthetic_images import evaluate_saved_model
from training_pipeline import training_pipeline_lfm
from model_latent_flow_matching import Encoder, Decoder, UNetSmall, LatentFlowMatching
import time


if __name__ == "__main__":
    training_pipeline_lfm(LatentFlowMatching, num_train_samples=50000, num_test_samples=10000, num_epochs=150, device='mps', batch_size=256)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(LatentFlowMatching, checkpoint_path='checkpoints/latent_flow_epoch_150.pth', test_size=10000, device='mps', num_visualize=15)
