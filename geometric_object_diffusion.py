import numpy as np
import svgwrite
from scipy.stats import pareto
import os
import torch
from mnist_experiments_jan import train_model, generate_with_model


def sample_geometric_object(h1=None, angle=None, scaling_factor=None, alpha=0.99):
    pareto_sample = pareto.rvs(alpha)
    uniform_sample = np.random.uniform(1, 5)
    h1 = pareto_sample if h1 is None else h1
    angle = np.random.uniform(0,  2 * np.pi) if angle is None else angle
    scaling_factor = 1  # np.random.uniform(0.1, 1.0) if scaling_factor is None else scaling_factor

    h2 = 1 / h1
    h1, h2 = np.log(h1 * scaling_factor + 1.5), np.log(h2 * scaling_factor + 1.5)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    points = np.array([[h1, -h1 / 4],
                       [h1, h1 / 4],
                       [-h2, -h2 / 4],
                       [-h2, h2 / 4]])
    points = points @ rotation_matrix.T
    return points


def save_geometric_object_svg(points, filename='geometric_object.svg', size=800):
    scaling_factor = size / 6
    points = points * scaling_factor
    center = np.array([size / 2, size / 2])
    dwg = svgwrite.Drawing(os.path.join("generated_images", filename), size=(size, size), profile='tiny')
    point_between_p1_p2 = (points[0] + points[1]) / 2
    point_between_p3_p4 = (points[2] + points[3]) / 2
    dwg.add(dwg.line(start=points[0] + center, end=points[1] + center, stroke=svgwrite.rgb(10, 10, 16, '%')))
    dwg.add(dwg.line(start=points[2] + center, end=points[3] + center, stroke=svgwrite.rgb(10, 10, 16, '%')))
    dwg.add(dwg.line(start=point_between_p1_p2 + center, end=point_between_p3_p4 + center, stroke=svgwrite.rgb(10, 10, 16, '%')))

    dwg.add(dwg.circle(center=center, r=2, fill='blue'))
    for point in points:
        dwg.add(dwg.circle(center=point + center, r=2, fill='red'))

    dwg.save()


def generate_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        points = sample_geometric_object()
        dataset.append(points)
    return np.array(dataset).reshape(num_samples, -1)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8 + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8)
        )

    def forward(self, x, t):
        t = t.view(-1, 1)  # Ensure t has shape [batch_size, 1]
        x_t = torch.cat([x, t], dim=1)  # Concatenate time embedding
        return self.network(x_t)


def analyze_generated_objects(generated_data):
    generated_data = generated_data.reshape(-1, 4, 2)

    scaling_factors = []
    h_values = []
    angles = []

    for i, points in enumerate(generated_data):
        between_p1_p2 = (points[0] + points[1]) / 2
        between_p3_p4 = (points[2] + points[3]) / 2
        p1_p2_length = np.linalg.norm(between_p1_p2)
        real_p1_p2_length = (np.exp(p1_p2_length) - 1.5)
        p3_p4_length = np.linalg.norm(between_p3_p4)
        real_p3_p4_length = (np.exp(p3_p4_length) - 1.5)

        scaling_factor_alternate = real_p1_p2_length * real_p3_p4_length
        h_alternate = real_p1_p2_length / scaling_factor_alternate
        h_values.append(h_alternate)
        scaling_factors.append(scaling_factor_alternate)

        print(f"Object {i}: h_alternate={h_alternate:.4f}, scaling_factor_alternate={scaling_factor_alternate:.4f}")

        vector_between = between_p1_p2 - between_p3_p4
        angle_vector_between = np.arctan2(vector_between[1], vector_between[0])
        angles.append(angle_vector_between)

    return np.array(h_values), np.array(scaling_factors), np.array(angles)


def plot_hs_against_pareto(h_values, alpha=0.99, min_limit=0, max_limit=15):
    import matplotlib.pyplot as plt
    from scipy.stats import pareto

    sorted_h = np.sort(h_values)
    empirical_cdf = np.arange(1, len(sorted_h) + 1) / len(sorted_h)

    x = np.linspace(min_limit, max_limit, 1000)
    pareto_cdf = pareto.cdf(x, alpha)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_h, empirical_cdf, label='Empirical CDF', marker='.', linestyle='none')
    plt.plot(x, pareto_cdf, label=f'Pareto CDF (alpha={alpha})', color='red')
    plt.xlabel('h values')
    plt.ylabel('CDF')
    plt.title('Empirical CDF of h values vs Pareto CDF')
    plt.xlim(left=min_limit, right=max_limit)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    dataset_length = 10000000
    dataset = generate_dataset(dataset_length)
    X_train = torch.Tensor(dataset[:int(0.8 * dataset_length)])
    X_test = torch.Tensor(dataset[int(0.8 * dataset_length):])

    model = Model()
    train_model(model, X_train, torch.tensor([0 for _ in range(len(X_train))]), X_test, torch.tensor([0 for _ in range(len(X_test))]), num_epochs=1, use_wandb=True, device='cpu', batch_size=32)
    X_test_generated = generate_with_model(model, num_samples=len(X_test), number_of_steps=50, device='cpu', start_noise=X_test)

    print("Generated dataset shape:", X_test_generated.shape)
    display_samples = 20
    for i in range(display_samples):
        points = X_test_generated[i].numpy().reshape(4, 2)
        save_geometric_object_svg(points, filename=f'generated_object_{i}.svg')

    h_values, scaling_factors, angles = analyze_generated_objects(X_test_generated.numpy())
    for i, (scaling_factor, h, angle) in enumerate(zip(scaling_factors[:display_samples], h_values[:display_samples], angles[:display_samples])):
        points = sample_geometric_object(h1=h, scaling_factor=scaling_factor, angle=angle)
        # save_geometric_object_svg(points, filename=f'analyzed_object_{i}.svg')
    plot_hs_against_pareto(h_values, alpha=0.99)
