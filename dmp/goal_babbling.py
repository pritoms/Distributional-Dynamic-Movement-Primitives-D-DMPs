import numpy as np
import torch
import pickle
import os
import glob

def save_trajectory(path, u, yd, yi, tau):
    """
    Save the trajectory to a pickle file.
    :param path: The path to save the trajectory to.
    :param u: The control inputs.
    :param yd: The desired trajectories.
    :param yi: The intention trajectories.
    :param tau: The time scaling.
    """
    with open(path, 'wb') as f:
        pickle.dump((u, yd, yi, tau), f)

def get_data(path):
    """
    Get the data from the pickle files.
    :param path: The path to the pickle files.
    :return: The control inputs and desired trajectories.
    """
    # Get all the pickle file paths
    paths = sorted(glob.glob('{0}/*.p'.format(path)))
    # Initialize the inputs and outputs lists
    inputs = []
    outputs = []
    # Run through the pickle files
    for p in paths:
        # Load in the data
        with open(p, 'rb') as f:
            u, yd, _, _ = pickle.load(f)
        # Add the control inputs and desired trajectories to the lists
        inputs.append(u)
        outputs.append(yd)
    return inputs, outputs

def generate_goal_babbling_trajectory(num_samples=100, num_timesteps=200, output_dim=2, tau=0.3):
    """
    Generate a goal babbling trajectory.
    :param num_samples: The number of samples in the trajectory.
    :param num_timesteps: The number of timesteps in the trajectory.
    :param output_dim: The dimensionality of the output.
    :param tau: The time scaling.
    :return: The control inputs, desired trajectories, intention trajectories, and time scaling.
    """
    # Set up the initial values
    u = np.zeros((num_samples, num_timesteps, output_dim))
    yd = np.zeros((num_samples, num_timesteps, output_dim))
    yi = np.zeros((num_samples, num_timesteps, output_dim))
    # Sample the desired end position from a unit ball centered at 0
    yf = np.random.multivariate_normal(np.zeros(output_dim), np.eye(output_dim), (num_samples, 1))
    # Assign the goal to be the desired end position
    u[:, num_timesteps - 1] = yf
    # Run through the number of samples
    for i in range(num_samples):
        # Generate a random time scaling
        tau_i = np.random.normal(tau, 0.05)
        # Run through the time steps
        for t in range(num_timesteps):
            # Update yd
            yd[i, t] = (1 - float(t) / num_timesteps) * yf[i]
            # Compute the intention for this timestep
            yi[i, t] = (yf[i] - yd[i, t]) / tau_i
    # Return the inputs, outputs, and tau
    return u, yd, yi, tau_i

def save_goal_babbling_trajectories(path, num_trajectories=100, num_samples=100, num_timesteps=200, output_dim=2, tau=0.3):
    """
    Save the goal babbling trajectories to pickle files.
    :param path: The path to save the trajectories to.
    :param num_trajectories: The number of trajectories to generate.
    :param num_samples: The number of samples in the trajectory.
    :param num_timesteps: The number of timesteps in the trajectory.
    :param output_dim: The dimensionality of the output.
    :param tau: The time scaling.
    """
    # Check if the path exists
    if not os.path.exists(path):
        # Create the path
        os.makedirs(path)
    # Run through the number of trajectories
    for i in range(num_trajectories):
        # Get the control inputs and desired trajectories
        u, yd, yi, tau_i = generate_goal_babbling_trajectory(num_samples, num_timesteps, output_dim, tau)
        # Save the trajectories
        save_trajectory('{0}/{1}.p'.format(path, i), u, yd, yi, tau_i)

def load_goal_babbling_trajectories(path):
    """
    Load the goal babbling trajectories from pickle files.
    :param path: The path to the pickle files.
    :return: The control inputs and desired trajectories.
    """
    # Get all the pickle file paths
    paths = sorted(glob.glob('{0}/*.p'.format(path)))
    # Initialize the inputs and outputs lists
    inputs = []
    outputs = []
    # Run through the pickle files
    for p in paths:
        # Load in the data
        with open(p, 'rb') as f:
            u, yd, _, _ = pickle.load(f)
        # Add the control inputs and desired trajectories to the lists
        inputs.append(u)
        outputs.append(yd)
    return inputs, outputs
