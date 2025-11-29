import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)

# 手动实现直方图均衡化
def manual_hist_equalization(image):
    # 计算图像的直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累计分布函数(CDF)
    cdf = hist.cumsum()

    # 归一化 CDF（映射到 0-255 范围）
    cdf_normalized = cdf * float(255) / cdf.max()

    # 使用归一化后的 CDF 映射原始图像的每个像素
    img_equalized = cdf_normalized[image]

    return img_equalized

# 应用手动直方图均衡化
equalized_image = manual_hist_equalization(image)

# 可视化原始图像和均衡化后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()
