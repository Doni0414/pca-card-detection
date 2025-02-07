### Task 9
# Finish the function below.
# Compute PCA over your dataset (data-1).

import glob
import numpy as np
import cv2


def task_9():
    DATA_DIR = 'data-2'
    TO_RESIZE = (64, int(64 * 0.625))

    data_filepaths = glob.glob('%s/*.jpeg' % DATA_DIR)

    images = []
    for filepath in data_filepaths:
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, TO_RESIZE)
        images.append(image)
    images = np.array(images)

     #######################
    # YOUR CODE GOES HERE #
    #######################

    N, H, W = images.shape

    X = images.reshape(N, H * W).copy()

    U = X.mean(axis=0)
    X_centered = X - U

    covariance_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[:,sorted_indexes]
   

    np.save('mean.npy', U)
    np.save('eigenvalues.npy', eigenvalues)
    np.save('eigenvectors.npy', eigenvectors)


if __name__ == '__main__':
    task_9()
