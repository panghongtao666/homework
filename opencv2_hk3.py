import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# 1. 高斯平滑（去噪声）
# ------------------------------
def gaussian_blur(img, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


# ------------------------------
# 2. 一阶差分计算梯度（修复：转为float32避免溢出）
# ------------------------------
def compute_gradients(img):
    # 关键修复1：将图像转为float32，避免uint8类型的负数溢出
    img_float = img.astype(np.float32)

    # 计算水平梯度 (I_x)
    I_x = np.zeros_like(img_float)
    I_x[:, 1:] = img_float[:, 1:] - img_float[:, :-1]  # 水平差分

    # 计算垂直梯度 (I_y)
    I_y = np.zeros_like(img_float)
    I_y[1:, :] = img_float[1:, :] - img_float[:-1, :]  # 垂直差分

    return I_x, I_y


# ------------------------------
# 3. 计算 I_x^2, I_y^2, I_x * I_y
# ------------------------------
def compute_squared_gradients(I_x, I_y):
    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y
    return I_x2, I_y2, I_xy


# ------------------------------
# 4. 高斯平滑 I_x^2, I_y^2, I_x * I_y
# ------------------------------
def gaussian_smooth(I_x2, I_y2, I_xy, window_size=5):
    S_x2 = cv2.GaussianBlur(I_x2, (window_size, window_size), 1.4)
    S_y2 = cv2.GaussianBlur(I_y2, (window_size, window_size), 1.4)
    S_xy = cv2.GaussianBlur(I_xy, (window_size, window_size), 1.4)
    return S_x2, S_y2, S_xy


# ------------------------------
# 5. 计算 Harris 响应 R
# ------------------------------
def harris_response(S_x2, S_y2, S_xy, alpha=0.04):
    det_M = S_x2 * S_y2 - S_xy ** 2  # 行列式
    trace_M = S_x2 + S_y2  # 迹
    R = det_M - alpha * trace_M ** 2
    return R


# ------------------------------
# 6. 角点检测与非极大值抑制（NMS）
# ------------------------------
def non_maximum_suppression(R, threshold=1e6):
    # 应用阈值（后续调用时会改小阈值）
    corners = np.zeros_like(R)
    corners[R > threshold] = 255

    # 使用 NMS 对角点进行精细化
    h, w = R.shape
    nms_corners = np.zeros_like(R)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if corners[i, j] == 255:
                # 检查邻域3x3中的最大值
                if np.max(corners[i - 1:i + 2, j - 1:j + 2]) == corners[i, j]:
                    nms_corners[i, j] = 255
                else:
                    nms_corners[i, j] = 0
    return nms_corners


# ------------------------------
# 主程序：读取图像 + 执行Harris角点检测
# ------------------------------
# 关键：修改为你的三角形图像路径（相对路径/绝对路径都可以）
img = cv2.imread("/home/pg/python练习/视觉/58132bc86370d_610.jpg")
if img is None:
    print("警告：未找到图像！请检查路径是否正确")
else:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    # 1. 高斯平滑
    blurred_img = gaussian_blur(img_gray)

    # 2. 计算梯度（已修复溢出问题）
    I_x, I_y = compute_gradients(blurred_img)

    # 3. 计算 I_x^2, I_y^2, I_x * I_y
    I_x2, I_y2, I_xy = compute_squared_gradients(I_x, I_y)

    # 4. 高斯平滑 I_x^2, I_y^2, I_x * I_y
    S_x2, S_y2, S_xy = gaussian_smooth(I_x2, I_y2, I_xy)

    # 5. 计算 Harris 响应 R
    R = harris_response(S_x2, S_y2, S_xy)

    # 6. 角点检测与NMS（关键修复2：降低阈值到1000，适配简单几何图形）
    corners = non_maximum_suppression(R, threshold=1000)

    # ------------------------------
    # 显示结果
    # ------------------------------
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("原图（灰度）")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Harris 角点检测结果")
    # 绘制红色角点（s=10：点的大小，可根据需求调整）
    plt.scatter(np.where(corners == 255)[1], np.where(corners == 255)[0], color='red', s=10)
    plt.axis('off')

    plt.tight_layout()
    plt.show()