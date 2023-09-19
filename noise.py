import numpy as np
import torch 
from numpy.random import choice 
from torch.utils.data import Dataset




def uniform_mix_C(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
           (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

class NoisyDatasetWrapper(Dataset):
    def __init__(self, base_dataset, noise_matrix):
        self.base_dataset = base_dataset
        self.noise_matrix = noise_matrix
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        noisy_label = choice(len(self.noise_matrix), 
                             p=self.noise_matrix[label])
        return img, torch.tensor(noisy_label)