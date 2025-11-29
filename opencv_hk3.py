import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 手写生成高斯核
def gaussian_kernel(size, sigma):
    center = (size - 1) / 2
    kernel = np.fromfunction(
        lambda x, y: np.exp(- ((x - center)**2 + (y - center)**2) / (2 * sigma**2)),
        (size, size)
    )
    kernel /= kernel.sum()
    return kernel
# 2. 手写卷积操作

def apply_filter(img, kernel):
    h, w = img.shape
    k = kernel.shape[0] // 2
    out = np.zeros_like(img)

    for i in range(k, h - k):
        for j in range(k, w - k):
            region = img[i - k:i + k + 1, j - k:j + k + 1]
            out[i, j] = np.sum(region * kernel)
    return out
# 3. 读取图像并测试不同参数

img = cv2.imread("/home/pg/python练习/视觉/image.png", 0)  # 替换为你的图片路径
#第一个参数这里需要实时更改路径
kernels = [
    gaussian_kernel(3, 1),
    gaussian_kernel(5, 1),
    gaussian_kernel(7, 2),
]

results = [apply_filter(img, k) for k in kernels]

# 4. 可视化结果
titles = ["3x3 σ=1", "5x5 σ=1", "7x7 σ=2"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

for i in range(3):
    plt.subplot(1, 4, i + 2)
    plt.imshow(results[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
