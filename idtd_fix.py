import cv2
import numpy as np

def new_ring_strel(ro, ri):
    d = 2 * ro + 1
    se = np.ones((d, d), dtype=np.uint8)
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0
    return se

def mnwth(img, delta_b, bb):
    img_f = img.copy()
    _, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    img_d = cv2.dilate(img, delta_b)
    img_e = cv2.erode(img_d, bb)
    out = cv2.subtract(img, img_e)
    out[out < 0] = 0
    return out

def smooth_image(frame):
    return cv2.GaussianBlur(frame, (7, 7), 0)

def move_detect(frame):
    ro = 11
    ri = 10
    delta_b = new_ring_strel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = mnwth(gray, delta_b, bb)
    return result

def detect_motion(frame):
    smoothed_frame = smooth_image(frame)
    result = move_detect(smoothed_frame)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc  # Returns the coordinates (x, y) of the detected motion