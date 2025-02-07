### Task 7
# Explain how rectangles are extracted from the given pairs of lines.


#### I WILL EXPLAIN IT IN THE FUNCTION   find_rectangles_from_line_intersections!!!
import cv2
import numpy as np
import random
from utils.geometry import find_orthogonal_lines, find_line_intersections, find_rectangles_from_line_intersections
from utils.image_processing import resize_image_keep_ratio, get_card_mask_for_image
from utils.canny import canny_edge_detection
from utils.hough import get_hough_lines

def task_7():
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

            # Probabilistic Hough Transform (please read)
            lines = get_hough_lines(edges, 60, 50, 10)

            # If there are lines and the number of lines is less than the given number
            if lines is not None and lines.shape[0] < 300:

                # Find orthogonal lines
                I, J = find_orthogonal_lines(lines)

                # Find line intersections
                x_points, x_pairs = find_line_intersections(lines, I, J, (h, w))

                if x_points.shape[0] > 1:

                    # Find rectangles using line intersection points
                    rectangles = find_rectangles_from_line_intersections(lines, x_points, x_pairs)
                    for idx in range(rectangles.shape[0]):
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        v_1, v_2, v_3, v_4 = rectangles[idx][0], rectangles[idx][1], rectangles[idx][2], rectangles[idx][3]

                        for point in [v_1, v_2, v_3, v_4]:
                            cv2.circle(image, tuple(point.astype(int).tolist()), 3, color, 3)

            cv2.imshow('camera', image_original)
            cv2.imshow('image', image)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_7()
