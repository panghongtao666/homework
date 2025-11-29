import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image = cv2.imread('/home/pg/python练习/视觉/test.jpeg', cv2.IMREAD_GRAYSCALE)  # 转为灰度图像

# 2. 使用 Sobel 算子计算梯度
# Sobel 算子计算 x 和 y 方向的梯度
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x方向
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y方向

# 3. 计算梯度幅值
magnitude = cv2.magnitude(sobel_x, sobel_y)

# 4. 可视化结果
plt.figure(figsize=(10, 7))

# 显示原图像
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# 显示 Sobel x 方向的结果
plt.subplot(1, 3, 2)
plt.title('Sobel X')
plt.imshow(sobel_x, cmap='gray')

# 显示梯度幅值
plt.subplot(1, 3, 3)
plt.title('Gradient Magnitude')
plt.imshow(magnitude, cmap='gray')

plt.tight_layout()
plt.show()
