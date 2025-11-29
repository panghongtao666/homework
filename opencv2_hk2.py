import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. 高斯平滑（去噪声）
# ------------------------------
def gaussian_blur(img, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

# ------------------------------
# 2. 计算梯度（Sobel 算子）
# ------------------------------
def gradient(img):
    # Sobel 水平和垂直方向的梯度
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值和方向
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    return magnitude, angle

# ------------------------------
# 3. 非极大值抑制（NMS）
# ------------------------------
def non_maximum_suppression(magnitude, angle):
    h, w = magnitude.shape
    nms_output = np.zeros_like(magnitude)

    angle = angle % 180  # 角度归一化到 0-180 之间

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            current_angle = angle[i, j]

            # 水平边缘
            if (current_angle >= 0 and current_angle < 22.5) or (current_angle >= 157.5 and current_angle < 180):
                neighbor1 = magnitude[i, j - 1]
                neighbor2 = magnitude[i, j + 1]
            # 垂直边缘
            elif (current_angle >= 22.5 and current_angle < 67.5):
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]
            # 45度角
            elif (current_angle >= 67.5 and current_angle < 112.5):
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]
            # 135度角
            elif (current_angle >= 112.5 and current_angle < 157.5):
                neighbor1 = magnitude[i - 1, j + 1]
                neighbor2 = magnitude[i + 1, j - 1]

            # 非极大值抑制
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                nms_output[i, j] = magnitude[i, j]
            else:
                nms_output[i, j] = 0

    return nms_output

# ------------------------------
# 4. 双阈值检测和边缘连接
# ------------------------------
def double_threshold(nms_result, low_threshold, high_threshold):
    h, w = nms_result.shape
    edges = np.zeros_like(nms_result)

    weak = 50  # 弱边缘值
    strong = 255  # 强边缘值

    # 强边缘
    edges[nms_result >= high_threshold] = strong
    # 弱边缘
    edges[(nms_result >= low_threshold) & (nms_result < high_threshold)] = weak

    # 边缘连接：连接弱边缘和强边缘
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if edges[i, j] == weak:
                if ((edges[i + 1, j - 1] == strong)
                        or (edges[i + 1, j] == strong)
                        or (edges[i + 1, j + 1] == strong)
                        or (edges[i, j - 1] == strong)
                        or (edges[i, j + 1] == strong)
                        or (edges[i - 1, j - 1] == strong)
                        or (edges[i - 1, j] == strong)
                        or (edges[i - 1, j + 1] == strong)):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0

    return edges

# ------------------------------
# 5. 显示每一步的结果
# ------------------------------
img = cv2.imread("/home/pg/python练习/视觉/test.jpeg")  # 读取原图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

# 1. 高斯平滑
blurred_img = gaussian_blur(img_gray)

# 2. 计算梯度（Sobel）
magnitude, angle = gradient(blurred_img)

# 3. 非极大值抑制（NMS）
nms_result = non_maximum_suppression(magnitude, angle)

# 4. 双阈值检测和边缘连接
low_threshold = 50
high_threshold = 150
final_edges = double_threshold(nms_result, low_threshold, high_threshold)

# ------------------------------
# 显示结果
# ------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(1, 4, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("原图（灰度）")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(blurred_img, cmap='gray')
plt.title("高斯平滑")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(nms_result, cmap='gray')
plt.title("NMS 结果")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(final_edges, cmap='gray')
plt.title("Canny 边缘")
plt.axis('off')

plt.show()
