# 理解利用了高斯核改进了AI的算法，也算是一个小小的进步
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 手动生成高斯核
def gaussian_kernel(size, sigma):
    center = (size - 1) // 2
    kernel = np.fromfunction(
        lambda x, y: np.exp(- ((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel /= kernel.sum()  # 归一化
    return kernel

# 2. 手动 Zero Padding（填充 0）
def zero_padding(img, kernel):
    pad = kernel.shape[0] // 2  # 自动根据高斯核大小计算 pad
    h, w = img.shape
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)
    out[pad:pad + h, pad:pad + w] = img
    return out

# 3. 手动 Replicate Padding（复制边界）
def replicate_padding(img, kernel):
    pad = kernel.shape[0] // 2  # 自动根据高斯核大小计算 pad
    h, w = img.shape
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)

    out[pad:pad + h, pad:pad + w] = img   # 把中间填进去

    # 上下边界
    out[:pad, pad:pad + w] = img[0:1, :]          # 顶部复制第一行
    out[pad + h:, pad:pad + w] = img[-1:, :]      # 底部复制最后一行

    # 左右边界
    out[pad:pad + h, :pad] = img[:, 0:1]          # 左边复制第一列
    out[pad:pad + h, pad + w:] = img[:, -1:]      # 右边复制最后一列

    # 四个角
    out[:pad, :pad] = img[0, 0]
    out[:pad, pad + w:] = img[0, -1]
    out[pad + h:, :pad] = img[-1, 0]
    out[pad + h:, pad + w:] = img[-1, -1]

    return out
# 4. 加载图片
img = cv2.imread("/home/pg/python练习/视觉/test.jpeg", cv2.IMREAD_GRAYSCALE)  # 记得换成你自己的图片路径

# 5. 生成高斯核（比如 5x5, sigma=1）
kernel = gaussian_kernel(5, 1)  # 生成一个 5x5 的高斯核，σ=1


# 6. 使用 padding

zero_pad_img = zero_padding(img, kernel)
replicate_pad_img = replicate_padding(img, kernel)


# 7. 显示结果
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("原图")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(zero_pad_img, cmap='gray')
plt.title("Zero Padding")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(replicate_pad_img, cmap='gray')
plt.title("Replicate Padding")
plt.axis('off')

plt.show()
