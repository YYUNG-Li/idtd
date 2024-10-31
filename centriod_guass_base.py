import cv2
import numpy as np
import time
import os

def new_ring_strel(ro, ri):
    d = 2 * ro + 1
    se = np.ones((d, d), dtype=np.uint8)
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0
    return se

def mnwth(img, delta_b, bb):
    img_d = cv2.dilate(img, delta_b)
    img_e = cv2.erode(img_d, bb)
    out = cv2.subtract(img, img_e)
    out[out < 0] = 0
    return out

def smooth_image(frame):
    return cv2.GaussianBlur(frame, (7, 7), 0)
    # return cv2.medianBlur(frame, 5)  # 使用中值滤波
def move_detect(frame):
    ro = 11
    ri = 10
    delta_b = new_ring_strel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    result = mnwth(gray, delta_b, bb)
    return result


def get_subpixel_centroid_using_minmax(result, frame):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # 获取最小值和最大值的位置
    cx, cy = max_loc  # 默认使用最大值位置

    search_window = 3
    region = result[cy - search_window:cy + search_window + 1, cx - search_window:cx + search_window + 1]

    total = region.sum()
    if total == 0:
        return (float('nan'), float('nan'))  # 如果区域总和为0，则返回NaN

    # 创建一个网格，用于计算亚像素质心
    h, w = region.shape
    y_indices, x_indices = np.indices((h, w))

    # 计算亚像素级质心
    cx = (region * x_indices).sum() / total + (cx - search_window)
    cy = (region * y_indices).sum() / total + (cy - search_window)

    return (cx, cy)

def get_subpixel_centroid_using_gradient(result, frame):
    # 将输入图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子计算x方向和y方向的梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # x方向梯度
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # y方向梯度

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 计算梯度幅值的标准差
    std_dev = cv2.meanStdDev(gradient_magnitude)[1][0][0]
    threshold = std_dev * 0.5  # 设定阈值为标准差的一半

    # 找出所有梯度幅值大于阈值的候选点
    candidates = np.where(gradient_magnitude > threshold)

    # 初始化质心坐标为NaN
    cx, cy = float('nan'), float('nan')

    # 如果找到候选点，则提取最大梯度位置作为质心
    if len(candidates[0]) > 0:
        # 使用加权平均计算质心位置
        weights = gradient_magnitude[candidates]
        cx = np.average(candidates[1], weights=weights)
        cy = np.average(candidates[0], weights=weights)
        print(f"Using gradient-based location: ({cx}, {cy})")  # 输出选择的质心位置
    else:
        print("No candidates found for centroid.")  # 未找到合适的候选点

    return (cx, cy)  # 返回计算得到的质心坐标



def show_img(frame, start_time):
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

        centroid = get_subpixel_centroid_using_minmax(result, frame)  # 获取亚像素质心

        cx, cy = centroid

        if np.isnan(cx) or (cx == 0 and cy == 0):
            print(f"Centroid not found or is (0,0), using gradient-based method for image {image_file}.")
            cx, cy = get_subpixel_centroid_using_gradient(result, frame)  # 基于梯度选择质心


        # 绘制质心点
        if not np.isnan(cx) and not np.isnan(cy):
            cv2.circle(frame, (int(cx), int(cy)), 1, (0, 255, 0), -1)
            show_img(frame, start_time)



        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")



    cv2.destroyAllWindows()
