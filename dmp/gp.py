import torch
import numpy as np
from torch import nn

class GaussianProcess(nn.Module):
    """
    Gaussian Process Regression
    """
    def __init__(self, input_dim, output_dim, basis_stddev=0.5, num_basis=20, dtype=torch.float64, device='cpu'):
        """
        Initialize the Gaussian Process Regression model
        :param input_dim: The input dimension
        :param output_dim: The output dimension
        :param basis_stddev: The standard deviation of the basis functions
        :param num_basis: The number of basis functions
        :param dtype: The data type used for computations
        :param device: The device used for computations
        """
        # Call the parent init function
        super(GaussianProcess, self).__init__()
        # Set the input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Set the number of basis functions
        self.num_basis = num_basis
        # Set the standard deviation of the basis functions
        self.stddev = basis_stddev
        # Set the type and device used for computations
        self.dtype = dtype
        self.device = device
        # Create the basis functions
        self.num_timesteps = 0
        self.basis = None
        self.create_basis()
        # Initialize the weights
        self.w = torch.nn.Parameter(torch.zeros(self.output_dim, self.num_basis).type(self.dtype))

    def create_basis(self):
        """
        Create the basis functions for the Gaussian Process
        """
        # Create the basis function for this GP
        self.basis = torch.zeros((1, self.input_dim, self.num_basis)).type(self.dtype)
        # Run through the number of basis functions
        for i in range(self.num_basis):
            # Set this basis function to be a 1-D Gaussian centered at a random location between 0 and 1
            mu = np.random.rand()
            self.basis[0, :, i] = torch.tensor(gaussian_pdf(np.linspace(0, 1, self.input_dim), mu=mu, sigma=self.stddev)).type(self.dtype)

    def forward(self, x):
        """
        Compute the Gaussian Process Regression
        :param x: The input data
        :return: The output of the Gaussian Process Regression
        """
        # Check if we have a single input dimension
        if len(x.shape) == 2:
            # Reshape the inputs
            x = x.view(-1, 1, self.input_dim)
        # Get the number of timesteps
        num_timesteps = x.shape[1]
        # Check if we need to update the basis function
        if num_timesteps > self.num_timesteps:
            # Update the number of timesteps
            self.num_timesteps = num_timesteps
            # Expand the basis functions
            self.basis = torch.zeros((num_timesteps, self.input_dim, self.num_basis)).type(self.dtype)
            for i in range(self.num_basis):
                mu = np.random.rand()
                self.basis[:, :, i] = torch.tensor(gaussian_pdf(np.linspace(0, 1, self.input_dim), mu=mu, sigma=self.stddev)).type(self.dtype)
        # Broadcast to compute the Gaussian process across all timesteps
        x = x.unsqueeze(-1) * self.basis
        # Return the linear combination of the basis functions
        return x @ torch.transpose(self.w, 0, 1)

    def mu(self, t):
        """
        Compute the mean of the Gaussian Process
        :param t: The timestep
        :return: The mean of the Gaussian Process
        """
        # Compute the mean for each basis function
        mean = torch.sum(torch.expand_dims(self.basis[t, :], dim=-1) * torch.expand_dims(self.w, dim=0), dim=1)
        # Return the means
        return mean

    def sigma(self, t):
        """
        Compute the standard deviation of the Gaussian Process
        :param t: The timestep
        :return: The standard deviation of the Gaussian Process
        """
        # Compute the variance for each basis function
        variance = torch.sum(torch.expand_dims(self.basis[t, :]**2, dim=-1) * torch.expand_dims(self.w**2, dim=0) - self.mu(t)**2, dim=1)
        # Return the standard deviations
        return torch.sqrt(variance)
