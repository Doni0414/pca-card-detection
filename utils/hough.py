import cv2
import numpy as np


def get_hough_lines(edges, threshold, line_min_length, line_max_gap):
    lines = cv2.HoughLinesP(edges, theta=np.pi/180, rho=1,threshold=threshold, minLineLength=line_min_length, maxLineGap=line_max_gap)

    #######################
    # YOUR CODE GOES HERE #
    #######################

    return lines
