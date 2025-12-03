from synthetic_images import evaluate_saved_model
from training_pipeline import training_pipeline_nf
from model_normalizing_flow import NormalizingFlow
import time


if __name__ == "__main__":
    training_pipeline_nf(NormalizingFlow, num_train_samples=50000, num_test_samples=10000, num_epochs=150, device='mps', batch_size=256)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(NormalizingFlow, checkpoint_path='checkpoints/normalizing_flow_epoch_150.pth', test_size=10000, device='mps', num_visualize=15)
