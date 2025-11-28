import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from unet import UNetMedium, UNetSmall
import os
from model_abstract import Model


class FlowMatching(Model):
    def __init__(self, model_type='small', load_from_path=None, dimensionality=2):
        self.model_type = model_type
        self.dimensionality = dimensionality

        if model_type == 'small':
            self.model = UNetSmall(load_from_path=load_from_path)
        else:
            self.model = UNetMedium(load_from_path=load_from_path)

    def train(self, X_train, y_train, X_test, y_test, num_epochs=1, use_wandb=True, 
              device='cpu', batch_size=32, checkpoint_dir='checkpoints'):
        if use_wandb:
            wandb.init(project="mnist-diffusion", name=f"unet-{self.model_type}-mse-loss")
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=2e-5)

        train_data_loader, test_data_loader, loss_function = self._prepare_training(X_train, y_train, X_test, y_test, checkpoint_dir, batch_size, device)

        for epoch in tqdm(range(num_epochs)):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self.model.train()
            losses = []
            for X_batch, y_batch in tqdm(train_data_loader):
                optimizer.zero_grad()

                time = torch.rand(X_batch.shape[0]).reshape(
                    (-1, 1, 1) if self.dimensionality == 2 else (-1, 1)
                ).to(device)

                pure_noise_images = torch.randn(X_batch.shape).to(device)
                interpolated_images = time * X_batch + (1 - time) * pure_noise_images

                # Flow matching: predict the velocity (data - noise)
                velocity_target = X_batch - pure_noise_images

                predicted_velocity = self.model(interpolated_images, time)
                loss = loss_function(predicted_velocity, velocity_target)
                losses.append(loss.item())
                if use_wandb:
                    wandb.log({"train_loss": loss.item()})
                loss.backward()

                optimizer.step()

            avg_train_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")

            self.model.eval()
            with torch.no_grad():
                losses = []
                for next_test_batch in test_data_loader:
                    X_test_batch, y_test_batch = next_test_batch
                    time_test = torch.rand(X_test_batch.shape[0]).reshape(
                        (-1, 1, 1) if self.dimensionality == 2 else (-1, 1)
                    ).to(device)
                    pure_noise_test_images = torch.randn(X_test_batch.shape).to(device)
                    interpolated_test_images = time_test * X_test_batch + (1 - time_test) * pure_noise_test_images

                    # Flow matching: predict the velocity
                    velocity_target_test = X_test_batch - pure_noise_test_images
                    predicted_velocity_test = self.model(interpolated_test_images, time_test)
                    test_loss = loss_function(predicted_velocity_test, velocity_target_test)
                    losses.append(test_loss.item())
                avg_test_loss = sum(losses) / len(losses)
                print(f"After Epoch {epoch+1}, Test Loss: {avg_test_loss}")

            if use_wandb:
                wandb.log({"avg_train_loss_epoch": avg_train_loss, "avg_test_loss_epoch": avg_test_loss})

            # Save model checkpoint
            torch.save(self.model.state_dict(), f"{checkpoint_dir}/unet_{self.model_type}_epoch_{epoch+1}.pth")

        if use_wandb:
            wandb.finish()

        self.model.to('cpu')

    def generate(self, num_samples=5, number_of_steps=100, device='cpu', start_noise=None, sample_shape=(28, 28)):
        if start_noise is None:
            start_noise = torch.randn(num_samples, *sample_shape)

        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            # Start from pure noise at t=0
            generated_images = start_noise.to(device)
            dt = 1.0 / number_of_steps

            # Integrate from t=0 to t=1 (noise to data)
            for step in tqdm(range(number_of_steps)):
                time = torch.full(
                    (num_samples, 1, 1) if self.dimensionality == 2 else (num_samples, 1), 
                    step / number_of_steps
                ).to(device)
                # Predict velocity at current position
                velocity = self.model(generated_images, time)
                # Euler integration: move along the flow
                generated_images = generated_images + velocity * dt

        self.model.to('cpu')

        # Clip the image to valid range [0, 1]
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        return generated_images.to('cpu')

    def generate_dataset(self, num_samples, number_of_steps=100, device='cpu', 
                         max_images_per_batch=2048, sample_shape=(28, 28)):
        generated_images = []

        for batch_start in tqdm(range(0, num_samples, max_images_per_batch)):
            batch_end = min(batch_start + max_images_per_batch, num_samples)
            batch_size = batch_end - batch_start
            batch_images = self.generate(
                num_samples=batch_size, 
                number_of_steps=number_of_steps, 
                device=device,
                sample_shape=sample_shape
            )
            generated_images.append(batch_images)
        
        return torch.cat(generated_images, dim=0)
