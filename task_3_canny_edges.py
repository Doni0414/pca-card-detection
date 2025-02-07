### Task 3
# Finish the canny_edge_detection function.
# The function should apply Canny Edge Detection over BGR image and should return edges.
# Adjust hyperparameters accordingly (threshold 1, threshold 2, gaussian blur kernel size, gaussian blur sigma)

import cv2
import numpy as np
from utils.image_processing import resize_image_keep_ratio, get_card_mask_for_image
from utils.canny import canny_edge_detection


def task_3():
    MAX_HEIGHT = 500
    MAX_WIDTH = 500
    CARD_RATIO = 0.625

    cam = cv2.VideoCapture(0)
    while True:

        # Read image from camera
        _, image = cam.read()

        if image is not None:

            # Resize image to max height or max width
            image = resize_image_keep_ratio(image, MAX_HEIGHT, MAX_WIDTH)
            image_original = image.copy()

            # Get image size
            h, w, _ = image.shape

            # Mask the image
            mask = (get_card_mask_for_image(image, CARD_RATIO, 0.8) * 255).astype(np.uint8)
            mask_blurred = cv2.GaussianBlur(mask, (21, 21), 10)
            image = np.round(image * (mask_blurred / 255)).astype(np.uint8)
            
            # Canny edge detection
            edges = canny_edge_detection(image)

            cv2.imshow('camera', image_original)
            cv2.imshow('edges', edges)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_3()
