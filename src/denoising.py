import math
import torch
import gpytorch
import numpy as np
import pandas as pd

from pathlib import Path

# A simple helper class to define the GPR model with a specific kernel
class ExactGPModel(gpytorch.models.ExactGP):
    """
    A GPR model that combines a periodic, RBF, and Matern kernel to
    model complex, non-linear signals.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # The combination of kernels allows for modeling periodic, smooth, and rough signals
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
        ) + gpytorch.kernels.MaternKernel(nu=2.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def smooth_sarcomere_lengths_with_gpr(
    frames: np.ndarray,
    signal: np.ndarray,
    output_path: Path = None,
    min_noise: float = 1e-3,
    max_iter: int = 500,
    derivative_threshold: float = 0.00001,
    learning_rate: float = 0.02,
) -> np.ndarray:
    """
    Smoothes a sarcomere length signal over time using Gaussian Process Regression.
    
    This function internally standardizes the frames and mean-centers the signal before training.
    It returns only the smoothed signal.

    Args:
        signal (np.ndarray): The raw sarcomere length data (y-values).
        frames (np.ndarray): The corresponding frame indices (x-values).
        output_path (Path, optional): If provided, the smoothed signal will be saved to this path as a CSV file. Defaults to None.
        min_noise (float): Minimum noise value to stop training.
        max_iter (int): Maximum number of training iterations.
        derivative_threshold (float): Training stops if the change in noise is below this value.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        np.ndarray: The smoothed signal (the mean of the GPR prediction) in its original scale.
    """
    if len(frames) < 2:
        print("Warning: Not enough data points to perform GPR. Returning original signal.")
        return signal

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Standardize frames and mean-center the signal ---
    train_x = torch.from_numpy(frames).float().to(device)
    train_y = torch.from_numpy(signal).float().to(device)

    mean_y = torch.mean(train_y)
    y_centered = train_y - mean_y

    mean_x = torch.mean(train_x)
    std_x = torch.std(train_x)
    if std_x == 0:
        print("Warning: Standard deviation of frames is zero. Cannot standardize. Returning original signal.")
        return signal
    
    x_normed = (train_x - mean_x) / std_x

    mean_y = torch.mean(train_y)
    y_centered = train_y - mean_y

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(x_normed, y_centered, likelihood).to(device)
    
    likelihood.train()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    iter_num = 1
    noise_val = float('inf')
    noise_diff = float('inf')
    while noise_diff > derivative_threshold and noise_val > min_noise and iter_num <= max_iter:
        optimizer.zero_grad()
        output = model(x_normed)
        loss = -mll(output, y_centered)
        loss.backward()

        new_noise_val = model.likelihood.noise.item()
        noise_diff = np.abs(noise_val - new_noise_val)
        noise_val = new_noise_val

        if iter_num % 20 == 0:
           print(f"Iter {iter_num} / {max_iter} - Loss: {loss.item():.3f} | Noise: {noise_val:.3f}")

        optimizer.step()
        iter_num += 1

    likelihood.eval()
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_normed))
        normalized_mean = observed_pred.mean.cpu()

    # Denormalize the predictions by adding the original mean back
    smoothed_signal = normalized_mean + mean_y

    # --- Save the results if an output path is provided ---
    if output_path:
        smoothed_df = pd.DataFrame({'frame': frames, 'length_smoothed': smoothed_signal.numpy()})
        smoothed_df.to_csv(output_path, index=False)
        print(f"Saved smoothed signal to {output_path}")

    return smoothed_signal.numpy()