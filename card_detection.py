import cv2
import numpy as np
from utils.geometry import find_orthogonal_lines, find_line_intersections, find_rectangles_from_line_intersections
from utils.perspective_transform import four_point_transform
from utils.similarity import pca_reconstruction_distance
from utils.image_processing import resize_image_keep_ratio, get_card_mask_for_image
from utils.canny import canny_edge_detection
from utils.hough import get_hough_lines


mean = np.load('mean.npy').astype(np.double)
eigenvectors = np.load('eigenvectors.npy').astype(np.double)
eigenvalues = np.load('eigenvalues.npy').astype(np.double)



eigenvectors[:, eigenvalues < 0] = eigenvectors[:, eigenvalues < 0] * -1
eigenvalues[eigenvalues < 0] = eigenvalues[eigenvalues < 0] * -1
eigenvalues_sorted_indices = np.argsort(eigenvalues)


def detect_card():
    MAX_HEIGHT = 500
    MAX_WIDTH = 500
    CARD_RATIO = 0.625
    PCA_MAX_DISTANCE = 2500
    PCA_FIRST_K = 60
    PCA_P_NORM = 2
    TO_RESIZE = (64, int(64 * 0.625))

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

                    image_original_for_copy = image_original.copy()

                    # Find best rectanbles using PCA
                    best = None
                    if rectangles.shape[0] > 0:
                        for idx in np.random.randint(rectangles.shape[0], size=2):
                            v_1, v_2, v_3, v_4 = rectangles[idx][0], rectangles[idx][1], rectangles[idx][2], rectangles[idx][3]
                            
                            warped_original = four_point_transform(image_original_for_copy, np.array([v_1, v_2, v_3, v_4]))
                            warped = cv2.resize(warped_original, TO_RESIZE)
                            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                            # Compute reconstruction distance
                            d = pca_reconstruction_distance(warped_gray, mean, eigenvectors, eigenvalues_sorted_indices, PCA_FIRST_K, PCA_P_NORM)
                            print('PCA distance: %s' % d)
                            if best is None:
                                best = ((v_1, v_2, v_3, v_4), warped_original, d)
                            
                            if d < best[2]:
                                best = ((v_1, v_2, v_3, v_4), warped_original, d)
                        
                        if best is not None:
                            if best[2] < PCA_MAX_DISTANCE:
                                cv2.imshow('found', best[1])
                                for point in best[0]:
                                    cv2.circle(image_original, tuple(point.astype(int).tolist()), 2, (0, 255, 0), 3)
                        

            # image_original = cv2.resize(image_original, (1280, 760))
            cv2.imshow('camera', image_original)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(mean)
    print(eigenvectors)
    print(eigenvalues)
    detect_card()
