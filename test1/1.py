import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# 配置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def normalize_band(band, min_val=0, max_val=10000):
    """将波段数据归一化到0-255范围"""
    band = np.clip(band, min_val, max_val)
    return ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def process_sentinel2(tif_file):
    """处理哨兵2号遥感影像并显示RGB图像"""
    with rasterio.open(tif_file) as src:
        bands = src.read()

        blue = bands[0].astype(float)  # B02 - 蓝
        green = bands[1].astype(float)  # B03 - 绿
        red = bands[2].astype(float)  # B04 - 红
        nir = bands[3].astype(float)  # B08 - 近红外
        swir = bands[4].astype(float)  # B12 - 短波红外

        # 原始图像（直接将DN值转换为0-255范围显示）
        # 注意：原始DN值通常远大于255，因此需要缩放
        # 这里使用简单的缩放方法，可能无法显示最佳效果
        red_raw = (red / 100).clip(0, 255).astype(np.uint8)
        green_raw = (green / 100).clip(0, 255).astype(np.uint8)
        blue_raw = (blue / 100).clip(0, 255).astype(np.uint8)
        rgb_raw = np.dstack((red_raw, green_raw, blue_raw))

        # 分别对每个通道进行归一化
        red_normalized = normalize_band(red)
        green_normalized = normalize_band(green)
        blue_normalized = normalize_band(blue)
        rgb_separate = np.dstack((red_normalized, green_normalized, blue_normalized))

        # 整体归一化
        rgb_original = np.dstack((red, green, blue))
        array_min, array_max = rgb_original.min(), rgb_original.max()
        rgb_normalized = ((rgb_original - array_min) / (array_max - array_min) * 255).astype(np.uint8)

        # 显示结果（添加原始图像作为对比）
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(rgb_raw)
        ax1.set_title('原始图像')
        ax1.axis('off')

        ax2.imshow(rgb_separate)
        ax2.set_title('独立通道归一化')
        ax2.axis('off')

        ax3.imshow(rgb_normalized)
        ax3.set_title('整体归一化')
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

        return rgb_raw, rgb_separate, rgb_normalized


# 示例调用
if __name__ == "__main__":
    file_path = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    rgb_raw, rgb_separate, rgb_normalized = process_sentinel2(file_path)