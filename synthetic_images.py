import numpy as np
from mnist_experiments_jan import visualize_n_samples, train_model, generate_with_model, UNetSmall, example_load_and_generate
import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from unet import UNetSmall, UNetMedium


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


def count_white_pixels(image_dataset):
    labels = []
    binary_images = (image_dataset > 0.5).float()
    for img in tqdm(binary_images):
        labels.append(int(torch.sum(img).item()))
    return labels


def generate_synthetic_dataset(num_samples):
    images = []
    labels = []
    for _ in tqdm(range(num_samples)):
        img, label = generate_synthetic_image()
        images.append(img)
        labels.append(label)
    return torch.tensor(np.array(images)), torch.tensor(np.array(labels))


def training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=512, use_wandb=True, unet_type='small'):
    X_train, y_train = generate_synthetic_dataset(num_train_samples)
    X_test, y_test = generate_synthetic_dataset(num_test_samples)

    # visualize_n_samples(X_test, y_test, n=15)

    model = UNetSmall() if unet_type == 'small' else UNetMedium()
    train_model(model, X_train, y_train, X_test, y_test, num_epochs=num_epochs, use_wandb=use_wandb, device=device, batch_size=batch_size, unet_type=unet_type)

    # generated_images = generate_with_model(model)
    # visualize_n_samples(generated_images, n=5)

    return model


def create_histogram(detected_counts, y_test):
    plt.figure()
    plt.hist(detected_counts, bins=range(min(detected_counts), max(detected_counts) + 2), alpha=0.5, label='Detected Counts')
    plt.hist(y_test.numpy(), bins=range(y_test.min().item(), y_test.max().item() + 2), alpha=0.5, label='True Counts')
    plt.title("Histogram of Detected Counts vs True Counts")
    plt.legend()
    plt.show()


def create_qq_plot(detected_counts, y_test):
    # Sort both datasets
    sorted_detected = np.sort(detected_counts)
    sorted_y_test = np.sort(y_test.numpy())

    # Create QQ plot
    plt.figure(figsize=(8, 8))
    plt.scatter(sorted_y_test, sorted_detected, alpha=0.5)

    # Add diagonal reference line
    min_val = min(sorted_y_test.min(), sorted_detected.min())
    max_val = max(sorted_y_test.max(), sorted_detected.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')

    plt.xlabel('True Counts (y_test)')
    plt.ylabel('Detected Counts')
    plt.title('QQ Plot: Detected Counts vs True Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def evaluate_saved_model(checkpoint_path, test_size=100000, device='mps', number_of_steps=25, num_visualize=15):
    X_test, y_test = generate_synthetic_dataset(test_size)
    visualize_n_samples(X_test, y_test, n=num_visualize)

    generated_images = example_load_and_generate(checkpoint_path, num_samples=test_size, device=device, number_of_steps=number_of_steps)
    detected_counts = count_white_pixels(generated_images)
    visualize_n_samples(generated_images, n=min(num_visualize, len(generated_images)), output_binarization=True, y_train=detected_counts)

    create_histogram(detected_counts, y_test)
    create_qq_plot(detected_counts, y_test)

    return generated_images, detected_counts, y_test


if __name__ == "__main__":
    training_pipeline(num_train_samples=500000, num_test_samples=100000, num_epochs=50, device='mps', batch_size=256, use_wandb=True, unet_type='medium')

    # evaluate_saved_model('checkpoints/unet_small_epoch_50.pth', test_size=100000, device='mps', number_of_steps=25)
