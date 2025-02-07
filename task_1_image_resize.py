### Task 1
# Finish the resize_image_keep_ratio function.
# The function should resize the given image to have the given max height or the given max width.
# The function must keep image height to width ratio.
# The function gets BGR image and returns resized BGR image.

import cv2
from utils.image_processing import resize_image_keep_ratio


def task_1():
    MAX_HEIGHT = 500
    MAX_WIDTH = 500

    cam = cv2.VideoCapture(0)
    while True:

        # Read image from camera
        _, image = cam.read()

        if image is not None:

            # Resize image to max height or max width
            image = resize_image_keep_ratio(image, MAX_HEIGHT, MAX_WIDTH)
            print(image.shape)

            cv2.imshow('camera', image)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_1()
