import cv2
import numpy as np
import time
import math
import os


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


def show_img(frame, start_time):
    # elapsed_time = (time.time() - start_time) * 1000
    cv2.imshow("frame", frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    image_folder = r'D:\private\lyy_data\idtd'
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.bmp'))]

    if not image_files:
        print("No images found in the specified folder.")
        exit()

    for image_file in image_files:
        start_time = time.time()
        frame = cv2.imread(os.path.join(image_folder, image_file))
        if frame is None:
            print(f"Failed to read image {image_file}.")
            continue



        smoothed_frame = smooth_image(frame)
        result = move_detect(smoothed_frame)

        # _, binary_img = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)
        #
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=4)
        # current_points = []
        # current_scale = []
        # for i in range(1, num_labels):
        #     x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
        #     y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
        #     w = stats[i, cv2.CC_STAT_WIDTH]
        #     h = stats[i, cv2.CC_STAT_HEIGHT]
        #     current_points.append((x, y))
        #     current_scale.append((w, h))

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc
        w, h = 20, 20
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
        show_img(frame, start_time)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

    cv2.destroyAllWindows()
