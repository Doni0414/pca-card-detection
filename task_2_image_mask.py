### Task 2
# Finish the get_card_mask_for_image function.
# The function should return a mask (2D matrix of float numbers [0.0, 1.0]) to hide unnecessary edges in the given image.
# The mask is made of 2 rectangles, where the outer rectangle spans all space vertically or horizontally and inner rectangle is downscaled by factor 0.9. 

import cv2
import numpy as np
from utils.image_processing import resize_image_keep_ratio, get_card_mask_for_image


def task_2():
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

            # Mask the image
            mask = (get_card_mask_for_image(image, CARD_RATIO, 0.9) * 255).astype(np.uint8)
            mask_blurred = cv2.GaussianBlur(mask, (21, 21), 10)
            image = np.round(image * (mask_blurred / 255)).astype(np.uint8)

            
            cv2.imshow('mask', mask)
            cv2.imshow('mask_blurred', mask_blurred)
            cv2.imshow('camera', image)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_2()
