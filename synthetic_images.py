import numpy as np
from mnist_experiments_jan import visualize_n_samples, train_model, generate_with_model, UNetSmall, example_load_and_generate
import torch
import matplotlib.pyplot as plt
import random


def check_for_availability(grid, p, orientation):
    if p[0] < 0 or p[0] >= grid.shape[0] or p[1] < 0 or p[1] >= grid.shape[1]:
        return False

    if orientation == 'horizontal':
        return (p[1] <= 0 or grid[p[0], p[1] - 1] == 0) and grid[p[0], p[1]] == 0 and (p[1] >= grid.shape[1] - 1 or grid[p[0], p[1] + 1] == 0)
    elif orientation == 'vertical':
        return (p[0] <= 0 or grid[p[0] - 1, p[1]] == 0) and grid[p[0], p[1]] == 0 and (p[0] >= grid.shape[0] - 1 or grid[p[0] + 1, p[1]] == 0)
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'")


def generate_synthetic_image():
    image = np.zeros((28, 28), dtype=np.float32)
    lrp = np.random.randint(0, 28, size=2)
    image[lrp[0], lrp[1]] = 1.0
    
    points_counter = 1
    target_point_counter = random.randint(1, 250)
    while points_counter < target_point_counter: # np.random.rand() < 0.98:
        available_directions = []
        if lrp[0] > 0 and check_for_availability(image, (lrp[0] - 1, lrp[1]), 'horizontal') and check_for_availability(image, (lrp[0] - 2, lrp[1]), 'horizontal'):
            available_directions.append('up')
        if lrp[0] < 27 and check_for_availability(image, (lrp[0] + 1, lrp[1]), 'horizontal') and check_for_availability(image, (lrp[0] + 2, lrp[1]), 'horizontal'):
            available_directions.append('down')
        if lrp[1] > 0 and check_for_availability(image, (lrp[0], lrp[1] - 1), 'vertical') and check_for_availability(image, (lrp[0], lrp[1] - 2), 'vertical'):
            available_directions.append('left')
        if lrp[1] < 27 and check_for_availability(image, (lrp[0], lrp[1] + 1), 'vertical') and check_for_availability(image, (lrp[0], lrp[1] + 2), 'vertical'):
            available_directions.append('right')
        if not available_directions:
            break

        direction = np.random.choice(available_directions)
        if direction == 'up' and lrp[0] > 0:
            lrp[0] -= 1
        elif direction == 'down' and lrp[0] < 27:
            lrp[0] += 1
        elif direction == 'left' and lrp[1] > 0:
            lrp[1] -= 1
        elif direction == 'right' and lrp[1] < 27:
            lrp[1] += 1
        image[lrp[0], lrp[1]] = 1.0
        points_counter += 1

    return image, points_counter


def generate_synthetic_dataset(num_samples):
    images = []
    labels = []
    for _ in range(num_samples):
        img, label = generate_synthetic_image()
        images.append(img)
        labels.append(label)
    return torch.tensor(images), torch.tensor(labels)


if __name__ == "__main__":
    X_train, y_train = generate_synthetic_dataset(500000)
    
    # Create a histogram over y_train
    plt.hist(y_train.numpy(), bins=range(y_train.min(), y_train.max() + 2))
    plt.title("Histogram of y_train")
    plt.show()

    X_test, y_test = generate_synthetic_dataset(100000)
    visualize_n_samples(X_train, y_train, n=15)
    model = UNetSmall()
    train_model(model, X_train, y_train, X_test, y_test, num_epochs=50, use_wandb=True, device='mps', batch_size=512)
    generated_images = generate_with_model(model)
    visualize_n_samples(generated_images, n=5)

    # Example: Load from checkpoint and generate
    generated_images = example_load_and_generate('checkpoints/unet_small_epoch_50.pth', num_samples=15, device='mps', number_of_steps=25)
    visualize_n_samples(generated_images, n=min(15, len(generated_images)), output_binarization=True)
