import glob
import pickle
import torch


class DataLoader(object):
    """
    This class loads in the data generated in Matlab and stores it as a torch tensor.
    """
    def __init__(self, path, shuffle=False):
        """
        Initialize a DataLoader object.
        Arguments:
            path (str): The path to where the data is stored.
            shuffle (bool): Whether or not to shuffle the data samples.
        """
        # Get all the pickle file paths
        self.paths = sorted(glob.glob('{0}/*.p'.format(path)))
        if shuffle:
            # Shuffle the paths
            np.random.shuffle(self.paths)
        self.num_samples = len(self.paths)

    def __getitem__(self, idx):
        """
        Get an item from the data loader.
        Arguments:
            idx (int): The index of the sample to get.
        Returns:
            tuple: A tuple containing the inputs, outputs, and intentions for this data point.
        """
        with open(self.paths[idx], 'rb') as f:
            # Load in the data
            u, yd, yi, tau = pickle.load(f)
        return torch.tensor(u).float(), torch.tensor(yd).float(), torch.tensor(yi).float()
