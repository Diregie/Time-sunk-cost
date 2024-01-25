import numpy as np


def cluster_count(x, eps):
    '''Calculates the number of clusters in HK-type models.

    Args:
        x (1xN numpy array): Opinions vector

        eps (float): The eps parameter of the model

    Returns:
        The number of clusters.
    '''
    if (len(x.shape) > 1):
        raise ValueError('Please provide a 1-D numpy array')

    sorted_opinions = np.sort(x)
    diffs = np.abs(np.diff(sorted_opinions))
    cluster_num = np.sum(diffs > eps/2) + 1
    return cluster_num