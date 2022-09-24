import os
import torch
import numpy as np
from torch.utils.data import Dataset

class DataLoader(Dataset):
    """
    This class is used to load the data generated using the Matlab implementation of D-DMPs.
    """
    def __init__(self, path, shuffle=True):
        """
        Initialize the DataLoader object.
        Arguments:
            path (str): The path to where the data is stored.
            shuffle (bool): Whether or not to shuffle the data.
        """
        self.path = path
        self.shuffle = shuffle

        # Get all the files in the given directory
        self.files = os.listdir(path)

        # Shuffle the files if necessary
        if self.shuffle:
            np.random.shuffle(self.files)

    def __len__(self):
        """
        Return the number of files in the given directory.
        Returns:
            int: The number of files in the given directory.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get the data for the given index.
        Arguments:
            idx (int): The index of the data to get.
        Returns:
            :obj:`torch.Tensor`: The input data.
            :obj:`torch.Tensor`: The output data.
            :obj:`torch.Tensor`: The intention data.
        """
        # Get the file name
        filename = self.files[idx]

        # Load the data
        data = np.load(os.path.join(self.path, filename))

        # Get the inputs, outputs, and intentions
        inputs = torch.from_numpy(data['inputs']).float()
        outputs = torch.from_numpy(data['outputs']).float()
        intentions = torch.from_numpy(data['intentions']).float()

        return inputs, outputs, intentions
