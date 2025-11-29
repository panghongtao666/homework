import numpy as np
import cv2
import matplotlib.pyplot as plt

# 手动实现二维高斯滤波核
def manual_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # 归一化
    return kernel

# OpenCV 生成高斯核
def opencv_gaussian_kernel(size, sigma):
    g1 = cv2.getGaussianKernel(size, sigma)
    return g1 @ g1.T  # 外积生成二维核

# 三组参数
sizes = [3, 5, 7]
sigmas = [0.5, 1.0, 2.0]

fig, axs = plt.subplots(len(sizes), 2, figsize=(10, 12))

for i in range(len(sizes)):
    size = sizes[i]
    sigma = sigmas[i]

    manual_kernel = manual_gaussian_kernel(size, sigma)
    cv_kernel = opencv_gaussian_kernel(size, sigma)

    axs[i, 0].imshow(manual_kernel, cmap='gray')
    axs[i, 0].set_title(f"手动高斯核 size={size}, σ={sigma}")

    axs[i, 1].imshow(cv_kernel, cmap='gray')
    axs[i, 1].set_title(f"OpenCV 高斯核 size={size}, σ={sigma}")

plt.tight_layout()
plt.show()
