#there are better ways to do this
#basically this works by looking for lines on the image and selecting the lines closest to
# the edge of the image, these are extrapolated to approximate where the corners are.
# Planing to significantly improve this
import numpy as np
import cv2
#cap = cv2.VideoCapture(0)

def get_src(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    rectangle = (1,1,638,478)

    cv2.grabCut(image, mask, rectangle,  
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    image_segmented = image * mask2[:, :, np.newaxis]

    # Assume `image_segmented` already exists from grabCut
    gray = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect straight lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=15, maxLineGap=10)

    line_img = image_segmented.copy()

    # Check if any lines were found
    for line in lines:
        x1, y1, x2, y2 = line[0]  # unpack the endpoints
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Hough Lines", line_img)

    left_horizontal = []
    top_vertical = []
    right_horizontal = []
    bottom_vertical = []
    hw = 300
    hh = 200
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2)/max(0.001,abs(x1 - x2)) < 0.5:  # horizontal line
            if x1 < hw or x2 < hw:
                left_horizontal.append(line[0])
            if x1 > hw or x2 > hw:
                right_horizontal.append(line[0])
        elif abs(y1 - y2)/max(0.001,abs(x1 - x2)) > 2:  # vertical line
            if y1 < hh or y2 < hh:
                bottom_vertical.append(line[0])
            if y1 > hh or y2 > hh:
                top_vertical.append(line[0])

    # Sort lines
    left_horizontal = sorted(left_horizontal, key=lambda l: l[1])
    top_vertical = sorted(top_vertical, key=lambda l: l[0])
    right_horizontal = sorted(right_horizontal, key=lambda l: l[1])
    bottom_vertical = sorted(bottom_vertical, key=lambda l: l[0])

    # Corners from intersections of grid extremes
    tl = line_intersection(left_horizontal[0], top_vertical[0])
    tr = line_intersection(right_horizontal[0], top_vertical[-1])
    bl = line_intersection(left_horizontal[-1], bottom_vertical[0])
    br = line_intersection(right_horizontal[-1], bottom_vertical[-1])

    corners = [tl, tr, br, bl]
    if any(c is None for c in corners):
        raise ValueError("Failed to detect all corners")

    src = np.array(corners, dtype='float32')
    return src

def get_average_src(img, cap):

    ret, img = cap.read()
    srcs = []
    for i in range(5):
        srcs.append(get_src(img))
    
    
    q =  0
    s = [[max([sublist[0][0] for sublist in srcs])-q,max([sublist[0][1] for sublist in srcs])-20],
         [min([sublist[1][0] for sublist in srcs])+q,max([sublist[1][1] for sublist in srcs])-20],
         [min([sublist[2][0] for sublist in srcs])+q,min([sublist[2][1] for sublist in srcs])+q],
         [max([sublist[3][0] for sublist in srcs])-q,min([sublist[3][1] for sublist in srcs])+q]]
    
    src = np.array(s)
    return src

def get_board(img,src):
    z = 50
   # src = srcs[0]#[max(a),max(b)],
    #src = get_average_src(img)
    dst = np.array([[0,0],[800,0],[800,800+z],[0,800+z]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(src, dst)
   # print(src)
    warped = cv2.warpPerspective(img, matrix, (800,800+z))
    return warped


# Get intersections
def line_intersection(h, v):
    x1, y1, x2, y2 = h
    x3, y3, x4, y4 = v
    A = np.array([[x2 - x1, x3 - x4],
                  [y2 - y1, y3 - y4]], dtype=float)
    b = np.array([x3 - x1, y3 - y1], dtype=float)
    if np.linalg.det(A) == 0:
        return None
    t = np.linalg.solve(A, b)
    px = x1 + t[0] * (x2 - x1)
    py = y1 + t[0] * (y2 - y1)
    return [int(px), int(py)]
