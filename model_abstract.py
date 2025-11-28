import os
import torch


class Model:
    def __init__(self, *args, **kwargs):
        print("Initializing abstract Model class.")

    def _prepare_training(self, X_train, y_train, X_test, y_test, checkpoint_dir, batch_size, device):
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device)), 
            batch_size=batch_size, shuffle=True
        )
        test_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device)), 
            batch_size=batch_size, shuffle=False
        )

        loss_function = torch.nn.MSELoss()

        return train_data_loader, test_data_loader, loss_function


    def train(self, X_train, y_train, X_test, y_test, num_epochs=1, use_wandb=True, 
              device='cpu', batch_size=32):
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def generate(self, num_samples, device='cpu', number_of_steps=25):
        raise NotImplementedError("Generate method must be implemented by subclasses.")

    def generate_dataset(self, num_samples, number_of_steps=100, device='cpu', 
                         max_images_per_batch=2048, sample_shape=(28, 28)):
        raise NotImplementedError("Generate dataset method must be implemented by subclasses.")
