import numpy as np

def find_line_equation(line):
    if len(line) == 2:
        x1, y1 = line[0]
        x2, y2 = line[1]
    else:
        x1, y1, x2, y2 = line
    
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1

    return k, b

def line_intersection(line1, line2):
    k1, b1 = find_line_equation(line1)
    k2, b2 = find_line_equation(line2)

    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    
    #######################
    # YOUR CODE GOES HERE #
    #######################

    return x, y


def find_orthogonal_lines(lines):
    I = []
    J = []
    #######################
    # YOUR CODE GOES HERE #
    #######################
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]

            angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0]) * 180 / np.pi
            angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0]) * 180 / np.pi

            angle_diff = np.abs(angle1 - angle2)

            if angle_diff > 80 and np.abs(angle_diff - 90) < 10:
                I.append(i)
                J.append(j)

    I = np.array(I)
    J = np.array(J)
    return I, J


def find_line_intersections(lines, I, J, inside):
    h, w = inside
    
    points = []
    pairs = []
    
    for idx in range(I.size):
        i = I[idx]
        j = J[idx]
        x, y = line_intersection(lines[i][0], lines[j][0])

        #######################
        # YOUR CODE GOES HERE #
        #######################

        if x >= 0 and x < w and y >= 0 and y < h:
            points.append((x, y))
            pairs.append((i, j))
    
    points = np.array(points)
    pairs = np.array(pairs)
    
    return points, pairs


def find_rectangles_from_line_intersections(lines, x_points, x_pairs, ratio=0.625, min_area=60000):
    C = x_points

    # x_pairs contains (i,j) tuple, such that ith and jth lines are intersecting to each other.
    # x_pairs[:,0] means we take all ith lines. Line is the array of (x1,y1,x2,y2) which means coordinates of the two points of the line
    # C2P1_1_X = lines[x_pairs[:,0], 0, 0] - C[:, 0] by this we make x coordinates - corresponding intersection by x axis with another line.
    # and so on for y1, x2, y2... 
    C2P1_1_X = lines[x_pairs[:,0], 0, 0] - C[:, 0]
    C2P1_1_Y = lines[x_pairs[:,0], 0, 1] - C[:, 1] 
    C2P1_1 = np.concatenate([C2P1_1_X[:, np.newaxis], C2P1_1_Y[:, np.newaxis]], axis=1)
    C2P1_2_X = lines[x_pairs[:,0], 0, 2] - C[:, 0]
    C2P1_2_Y = lines[x_pairs[:,0], 0, 3] - C[:, 1]
    C2P1_2 = np.concatenate([C2P1_2_X[:, np.newaxis], C2P1_2_Y[:, np.newaxis]], axis=1)

    #Compute distances or length of the C2P1_1 and C2P1_2
    D1 = np.linalg.norm(C2P1_1, axis=1)
    D2 = np.linalg.norm(C2P1_2, axis=1)

    C2P1 = np.zeros_like(C2P1_1)
    # here we save points that have longer distance than the second end points of the same line
    C2P1[D1 >= D2] = C2P1_1[D1 >= D2]
    C2P1[D1 < D2] = C2P1_2[D1 < D2]

    # this calculations are also the same but for the jth pair in the x_pairs that is (ith line, jth line)
    C2P2_1_X = lines[x_pairs[:,1], 0, 0] - C[:, 0]
    C2P2_1_Y = lines[x_pairs[:,1], 0, 1] - C[:, 1]
    C2P2_1 = np.concatenate([C2P2_1_X[:, np.newaxis], C2P2_1_Y[:, np.newaxis]], axis=1)
    C2P2_2_X = lines[x_pairs[:,1], 0, 2] - C[:, 0]
    C2P2_2_Y = lines[x_pairs[:,1], 0, 3] - C[:, 1]
    C2P2_2 = np.concatenate([C2P2_2_X[:, np.newaxis], C2P2_2_Y[:, np.newaxis]], axis=1)

    D1 = np.linalg.norm(C2P2_1, axis=1)
    D2 = np.linalg.norm(C2P2_2, axis=1)

    C2P2 = np.zeros_like(C2P2_1)

    C2P2[D1 >= D2] = C2P2_1[D1 >= D2]
    C2P2[D1 < D2] = C2P2_2[D1 < D2]

    # we make vector normalizations. So every vector will be unit vector, because they are divided by their length
    C_D = C2P1 / np.linalg.norm(C2P1, axis=1)[:, np.newaxis] + C2P2 / np.linalg.norm(C2P2, axis=1)[:, np.newaxis]
    C_D = C_D / np.linalg.norm(C_D, axis=1)[:, np.newaxis]

    # I don't mind what is done next :)
    A = C_D @ C_D.T
    
    I, J = np.where(np.abs((A + 1)) < 0.10)

    X_diff = np.abs(C[I, 0] - C[J, 0])
    Y_diff = np.abs(C[I, 1] - C[J, 1])

    R1 = X_diff / Y_diff
    R2 = Y_diff / X_diff
    area = X_diff * Y_diff

    F = np.where(np.logical_and(np.logical_or(np.abs((R1 - ratio)) < 0.10, np.abs((R2 - ratio)) < 0.10), area > min_area))

    vertex_1 = x_points[I[F],:]
    vertex_1_lines_1 = lines[x_pairs[I[F], 0],:,:] 
    vertex_1_lines_2 = lines[x_pairs[I[F], 1],:,:]
    vertex_1_to_p1 = C2P1[I[F],:]
    # vertex_1_to_p2 = C2P2[I[F],:]
    
    vertex_2 = x_points[J[F],:]
    vertex_2_lines_1 = lines[x_pairs[J[F], 0],:,:] 
    vertex_2_lines_2 = lines[x_pairs[J[F], 1],:,:]
    vertex_2_to_p1 = C2P1[J[F],:]
    # vertex_2_to_p2 = C2P2[J[F],:]

    rectangles = []
    for idx in range(vertex_1.shape[0]):
        # try:
            v_1 = vertex_1[idx]
            v_2 = vertex_2[idx]
            # compute cosine between vectors, to compute degree
            cos_v = np.matmul(vertex_1_to_p1[idx][:, np.newaxis].T, vertex_2_to_p1[idx][:, np.newaxis]) / (np.linalg.norm(vertex_1_to_p1[idx]) * np.linalg.norm(vertex_2_to_p1[idx]))
            
            if abs(cos_v) < 0.5:
                # v1_to_p1 and v2_to_p1 is orthogonal
                l11 = ((vertex_1_lines_1[idx][0][0], vertex_1_lines_1[idx][0][1]), (vertex_1_lines_1[idx][0][2], vertex_1_lines_1[idx][0][3]))
                l12 = ((vertex_2_lines_1[idx][0][0], vertex_2_lines_1[idx][0][1]), (vertex_2_lines_1[idx][0][2], vertex_2_lines_1[idx][0][3]))
                l21 = ((vertex_1_lines_2[idx][0][0], vertex_1_lines_2[idx][0][1]), (vertex_1_lines_2[idx][0][2], vertex_1_lines_2[idx][0][3]))
                l22 = ((vertex_2_lines_2[idx][0][0], vertex_2_lines_2[idx][0][1]), (vertex_2_lines_2[idx][0][2], vertex_2_lines_2[idx][0][3]))
            else:
                # v1_to_p1 and v2_to_p2 is orthogonal
                l11 = ((vertex_1_lines_1[idx][0][0], vertex_1_lines_1[idx][0][1]), (vertex_1_lines_1[idx][0][2], vertex_1_lines_1[idx][0][3]))
                l12 = ((vertex_2_lines_2[idx][0][0], vertex_2_lines_2[idx][0][1]), (vertex_2_lines_2[idx][0][2], vertex_2_lines_2[idx][0][3]))
                l21 = ((vertex_1_lines_2[idx][0][0], vertex_1_lines_2[idx][0][1]), (vertex_1_lines_2[idx][0][2], vertex_1_lines_2[idx][0][3]))
                l22 = ((vertex_2_lines_1[idx][0][0], vertex_2_lines_1[idx][0][1]), (vertex_2_lines_1[idx][0][2], vertex_2_lines_1[idx][0][3]))
            # find intersections of lines
            v_3 = np.array(line_intersection(l11, l12))
            v_4 = np.array(line_intersection(l21, l22))

            rectangles.append((v_1, v_2, v_3, v_4))

        # except Exception:
            # pass
    
    rectangles = np.array(rectangles)

    return rectangles
