import cv2


def canny_edge_detection(image, t1=50, t2=200, gaussian_blur_size=3, gaussian_blur_sigma=0):
    blurred = cv2.GaussianBlur(image, (gaussian_blur_size,gaussian_blur_size), gaussian_blur_sigma)
    edges = cv2.Canny(blurred, t1, t2)
    #######################
    # YOUR CODE GOES HERE #
    #######################
    
    return edges
