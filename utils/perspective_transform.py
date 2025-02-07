import cv2
import numpy as np

def distance(point1, point2):
	return np.sqrt(np.sum((point1 - point2)**2))

def order_points(pts):
	rect = np.zeros((4, 2), dtype=np.float32)
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	maxHeight = int(max(distance(br, tr), distance(bl, tl)))
	maxWidth = int(max(distance(tr, tl), distance(br, bl)))
	
	#######################
    # YOUR CODE GOES HERE #
    #######################

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

