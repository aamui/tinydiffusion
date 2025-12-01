from synthetic_images import evaluate_saved_model
from training_pipeline import training_pipeline
from model_flow_matching import FlowMatching
import time


if __name__ == "__main__":
    training_pipeline(FlowMatching, num_train_samples=50000, num_test_samples=10000, num_epochs=5, device='mps', batch_size=256, use_wandb=True, dimensionality=2, model_type='medium')

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(FlowMatching, checkpoint_path='checkpoints/unet_medium_epoch_5.pth', test_size=10000, device='mps', num_visualize=15)
