### Task 4
# Finish the get_hough_lines function.
# The function should return lines using HoughLinesP (Probabilistic Hough Transform).
# Adjust hyperparameters accordingly (threshold, line min length, line max gap).

import cv2
import numpy as np
import random
from utils.image_processing import resize_image_keep_ratio, get_card_mask_for_image
from utils.canny import canny_edge_detection
from utils.hough import get_hough_lines


def task_4():
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

            # Mask the image
            mask = (get_card_mask_for_image(image, CARD_RATIO, 0.8) * 255).astype(np.uint8)
            mask_blurred = cv2.GaussianBlur(mask, (21, 21), 10)
            image = np.round(image * (mask_blurred / 255)).astype(np.uint8)
            
            # Canny edge detection
            edges = canny_edge_detection(image)

            # Probabilistic Hough Transform (please read)
            lines = get_hough_lines(edges, 60, 50, 10)

            if lines is not None:
                for idx in range(lines.shape[0]):
                    x1, y1, x2, y2 = int(lines[idx][0][0]), int(lines[idx][0][1]), int(lines[idx][0][2]), int(lines[idx][0][3])
                    cv2.line(image, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)

            cv2.imshow('camera', image_original)
            cv2.imshow('image', image)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_4()
