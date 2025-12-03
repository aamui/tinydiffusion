from synthetic_images import evaluate_saved_model
from training_pipeline import training_pipeline_nf
from model_normalizing_flow2 import NormalizingFlow
import time
import torch
from synthetic_images import generate_synthetic_dataset, visualize_n_samples

if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"


# if __name__ == "__main__":
#     device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
#     num_train_samples=500000
#     num_test_samples=100000
#     X_train, y_train = generate_synthetic_dataset(num_train_samples, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_train.pt", normalize=True )
#     X_test, y_test = generate_synthetic_dataset(num_test_samples, use_saved=True, save_path="./data/synthetic/syntetic_data_normal_test.pt", normalize=True)

#     visualize_n_samples(X_test, y_test, n=15, file_name="test_samples.pdf")
   
#     num_classes = int(y_train.max().item() + 1)
#     X_train = X_train.view(X_train.size(0), -1)
#     X_test  = X_test.view(X_test.size(0), -1)
#     dim = X_train.size(1)

#     num_epochs = 150
#     nf_model = NormalizingFlow(dim=dim, num_classes=num_classes)

#     nf_model.train_function(X_train, y_train,X_test, y_test,num_epochs=num_epochs,device=device,model_name="syn_nf")

#     class_counts = torch.bincount(y_train, minlength=num_classes).float()
#     class_probs = class_counts / class_counts.sum()

#     nf_model = NormalizingFlow(dim=dim,num_classes=num_classes,load_from_path=f"./checkpoints/syn_nf_epoch_{num_epochs}.pth")

#     num_samples = 16
#     samples = nf_model.generate_dataset(num_samples=num_samples, device=device, class_probs=class_probs)
#     samples_img = samples.view(num_samples, 1, 28, 28)

#     visualize_n_samples(samples_img, n=5, output_binarization=True, file_name="syn_nf_generated_samples.pdf")

if __name__ == "__main__":
    print(f"Using device: {device}")
    training_pipeline_nf(NormalizingFlow, num_epochs=100, device=device, batch_size=256)

    print("Training completed, waiting before evaluation...")
    time.sleep(2)
    print("Starting evaluation...")

    evaluate_saved_model(NormalizingFlow, checkpoint_path='checkpoints/eval_syn_nf_epoch_10.pth', test_size=10000, device=device, num_visualize=15)
