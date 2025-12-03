import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from experiment_mnist_flow_matching import visualize_n_samples
import os
from scipy.stats import ks_2samp
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

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


def count_white_pixels(image_dataset, threshold = 0.5):
    labels = []
    binary_images = (image_dataset > threshold).float()
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

def generate_synthetic_dataset_channel_normalize(num_samples):
    images = []
    labels = []
    for _ in tqdm(range(num_samples)):
        img, label = generate_synthetic_image()
        images.append(img)
        labels.append(label)
    img = torch.tensor(np.array(images)).unsqueeze(1)
    img = (img - 0.5) / 0.5
    return img, torch.tensor(np.array(labels))

def generate_synthetic_dataset(num_samples, use_saved=False, save_path="synthetic_dataset.pt", normalize=False):
    if use_saved and os.path.exists(save_path):
        print(f"Loading saved dataset from {save_path}")
        data = torch.load(save_path)
        return data["images"], data["labels"]

    images = []
    labels = []

    for _ in tqdm(range(num_samples)):
        img, label = generate_synthetic_image()
        images.append(img)
        labels.append(label)

    images = torch.tensor(np.array(images))    
    labels = torch.tensor(np.array(labels))    

    if normalize:  
        images = (images - 0.5) / 0.5     


    torch.save({"images": images, "labels": labels}, save_path)
    print(f"Saved dataset to {save_path}")

    return images, labels

def create_histogram(detected_counts, y_test):
    plt.figure()
    plt.hist(detected_counts, bins=range(min(detected_counts), max(detected_counts) + 2), alpha=0.5, label='Detected Counts')
    plt.hist(y_test.numpy(), bins=range(y_test.min().item(), y_test.max().item() + 2), alpha=0.5, label='True Counts')
    plt.title("Histogram of Detected Counts vs True Counts")
    plt.legend()
    plt.savefig("plots/histogram_detected_vs_true_counts.pdf")
    plt.clf()
    plt.close()


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
    plt.savefig(f"plots/qq_plot_detected_vs_true_counts.pdf")
    plt.clf()
    plt.close()


def evaluate_saved_model(model_class, checkpoint_path, test_size=100000, device='mps', num_visualize=15):
    X_test, y_test = generate_synthetic_dataset(test_size)
    visualize_n_samples(X_test, y_test, n=num_visualize, file_name=f"test_samples_{model_class.__name__}.pdf")

    model = model_class(load_from_path=checkpoint_path)
    generated_images = model.generate_dataset(num_samples=test_size, device=device)
    detected_counts = count_white_pixels(generated_images)
    visualize_n_samples(generated_images, n=min(num_visualize, len(generated_images)), output_binarization=True, y_train=detected_counts, file_name=f"generated_samples_{model_class.__name__}.pdf")

    create_histogram(detected_counts, y_test)
    create_qq_plot(detected_counts, y_test)

    real_data = y_test.numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
    fake_data = np.array(detected_counts)

    ks_stat, p_value = ks_2samp(real_data, fake_data)

    print(f"--- KS Test Results ---")
    print(f"Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    if ks_stat < 0.05:
        print("PASS: Model captures the distribution well.")
    else:
        print("FAIL: Significant difference in distributions.")

    return generated_images, detected_counts, y_test

def sample_from_model(model_class, checkpoint_path, samples = 100, device = 'mps'):
    model = model_class(load_from_path = checkpoint_path)
    generated_images = model.generate_dataset(num_samples = samples, device = device)
    generated_images = (generated_images > 0.5).float()
    samples_path = f'samples/{model_class.__name__}_samples.pt'
    torch.save(generated_images, samples_path)
    print(f'Samples saved to {samples_path}')

def count_robust_worm_lengths(image_dataset, threshold=0.2, min_part_size=5):
    """
    1. Threshold low (0.2) to bridge gaps in the worm.
    2. Skeletonize to fix thickness.
    3. Remove 'dust' (components < 5 pixels).
    4. Sum the remaining lengths.
    """
    robust_counts = []
    
    # Convert to numpy if needed
    if isinstance(image_dataset, torch.Tensor):
        image_dataset = image_dataset.cpu().numpy()
        
    for img in image_dataset:
        # 1. Low Threshold to connect broken parts
        binary = img > threshold
        
        # 2. Skeletonize FIRST (Fix thickness before measuring)
        # Note: Skeletonize works on the full image, preserving connections
        skeleton = skeletonize(binary)
        
        # 3. Label connected regions
        labeled_img = label(skeleton, connectivity=2)
        regions = regionprops(labeled_img)
        
        total_length = 0
        for region in regions:
            # 4. Filter out dust (keep anything that looks like a worm segment)
            if region.area >= min_part_size:
                total_length += region.area
                
        robust_counts.append(int(total_length))
        
    return robust_counts

def evaluate_saved_model_threshold(model_class, checkpoint_path, test_size=100000, device='mps', num_visualize=15, threshold = 0.5):
    X_test, y_test = generate_synthetic_dataset(test_size)
    visualize_n_samples(X_test, y_test, n=num_visualize, file_name=f"test_samples_{model_class.__name__}_{threshold}.pdf", threshold = threshold)

    model = model_class(load_from_path=checkpoint_path)
    generated_images = model.generate_dataset(num_samples=test_size, device=device)
    detected_counts = count_white_pixels(generated_images, threshold = threshold)
    visualize_n_samples(generated_images, n=min(num_visualize, len(generated_images)), output_binarization=True, y_train=detected_counts, file_name=f"generated_samples_{model_class.__name__}_{threshold}.pdf", threshold = threshold)

    create_histogram(detected_counts, y_test)
    create_qq_plot(detected_counts, y_test)

    real_data = y_test.numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
    fake_data = np.array(detected_counts)

    ks_stat, p_value = ks_2samp(real_data, fake_data)

    print(f"--- KS Test Results ---")
    print(f"Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    if ks_stat < 0.05:
        print("PASS: Model captures the distribution well.")
    else:
        print("FAIL: Significant difference in distributions.")

    return generated_images, detected_counts, y_test