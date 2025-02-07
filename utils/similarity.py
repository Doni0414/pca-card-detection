import numpy as np


def minkowski_distance(a, b, p=2):
    return np.sum(np.abs(a - b) ** p)**(1/p)


def pca_reconstruction_distance(grayscale, mean, eigenvectors, eigenvalues_sorted_indices, k=1000, p=2):
    selected_indices = eigenvalues_sorted_indices[-k:]
    h, w = grayscale.shape[0], grayscale.shape[1]
    #######################
    # YOUR CODE GOES HERE #
    #######################
    v = grayscale.reshape(1, h * w)
    
  

    projection_matrix = eigenvectors[:,eigenvalues_sorted_indices]
    projection_matrix = projection_matrix[:,:k]

    projected_v = np.dot(v, projection_matrix)
    reconstructed = np.dot(projected_v, projection_matrix.T) + mean

    return minkowski_distance(v.squeeze(), reconstructed.squeeze(), p)
