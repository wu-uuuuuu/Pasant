import os
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.plot import show

# 设置中文字体以确保matplotlib能够正确显示中文
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def process_sentinel2_image(image_path, output_path=None, bands=None, show_image=True, visualize=True, use_gpu=True):
    """
    使用PyTorch处理哨兵2号遥感图像，将数据范围从0-10000压缩到0-255并转为RGB

    参数:
        image_path: 哨兵2号图像文件路径
        output_path: 输出图像保存路径，默认为None
        bands: 要使用的波段索引列表，默认为RGB波段(1,2,3)
        show_image: 是否显示处理后的图像，默认为True
        visualize: 是否显示原图和处理后的图像对比，默认为True
        use_gpu: 是否使用GPU加速，默认为True
    """
    # 1. 读取遥感图像数据
    try:
        with rasterio.open(image_path) as src:
            print(f"成功读取图像，波段数: {src.count}")
            print(f"数据范围: {src.meta['nodata']} 到 {np.nanmax(src.read())}")

            # 如果未指定波段，默认使用前三个波段作为RGB
            if bands is None:
                bands = [1, 2, 3]  # 假设前三个波段为BGR顺序，需要调整为RGB
                print(f"使用默认波段: {bands}")
            else:
                print(f"使用指定波段: {bands}")

            # 读取指定波段数据
            data = src.read(bands)

            # 转换为PyTorch张量
            device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            print(f"使用设备: {device}")

            # 将数据转换为PyTorch张量并移至指定设备
            # 修复：先将数据转换为float32类型
            tensor_data = torch.from_numpy(data.astype(np.float32)).to(device)

            # 2. 数据范围压缩处理 (0-10000 -> 0-255)
            # 处理前先替换无效值(nodata)为0
            nodata_value = src.meta.get('nodata', 0)
            tensor_data[tensor_data == nodata_value] = 0

            # 线性拉伸到0-255
            min_val, max_val = 0, 10000
            normalized_data = ((tensor_data - min_val) / (max_val - min_val) * 255).clamp(0, 255)

            # 转换为无符号8位整数，并移回CPU
            rgb_data = normalized_data.byte().cpu().numpy()

            # 3. 调整波段顺序为RGB (如果需要)
            # 假设输入波段顺序为BGR，转换为RGB
            if bands == [1, 2, 3]:  # 蓝、绿、红
                rgb_data = rgb_data[[2, 1, 0], :, :]  # 转换为红、绿、蓝

            # 4. 显示原图与处理后的图像对比
            if visualize:
                visualize_results(data, rgb_data, bands)

            # 5. 显示处理后的图像
            if show_image and not visualize:
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb_data.transpose(1, 2, 0))
                plt.title('Sentinel-2 RGB Visualization')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            # 6. 保存图像
            if output_path:
                # 确保输出路径是一个文件路径，而不是目录路径
                if os.path.isdir(output_path):
                    file_name = os.path.splitext(os.path.basename(image_path))[0] + '_rgb.jpg'
                    output_path = os.path.join(output_path, file_name)

                # 使用PIL保存图像
                img = Image.fromarray(rgb_data.transpose(1, 2, 0))
                img.save(output_path)
                print(f"图像已保存至: {output_path}")

            return rgb_data

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None


def visualize_results(original_data, processed_data, bands):
    """可视化原图与处理后的图像对比"""
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 获取波段名称
    band_names = {1: 'Blue', 2: 'Green', 3: 'Red', 4: 'NIR', 5: 'SWIR'}

    # 显示原图
    # 由于原始数据值较大，我们需要进行简单的归一化以便于显示
    original_display = np.zeros_like(processed_data, dtype=np.float32)
    for i in range(processed_data.shape[0]):
        band_min = np.min(original_data[i])
        band_max = np.max(original_data[i])
        if band_max > band_min:
            original_display[i] = (original_data[i] - band_min) / (band_max - band_min) * 255

    # 如果是RGB波段组合，调整通道顺序
    if bands == [1, 2, 3]:
        original_display = original_display[[2, 1, 0], :, :]

    axes[0].imshow(original_display.transpose(1, 2, 0).astype(np.uint8))
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 显示处理后的图像
    axes[1].imshow(processed_data.transpose(1, 2, 0))
    axes[1].set_title('处理后的图像 (0-255)')
    axes[1].axis('off')

    # 添加波段信息
    band_info = ', '.join([band_names.get(b, f'波段 {b}') for b in bands])
    plt.suptitle(f'波段组合: {band_info}', fontsize=16)

    plt.tight_layout()
    plt.show()


def batch_process_sentinel2(folder_path, output_folder, bands=None, show_image=False, visualize=False, use_gpu=True):
    """批量处理文件夹中的哨兵2号图像"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 查找所有可能的遥感图像文件
    image_extensions = ['.tif', '.TIF', '.tiff', '.TIFF']
    image_files = []

    for ext in image_extensions:
        image_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.endswith(ext)])

    if not image_files:
        print(f"在{folder_path}中未找到遥感图像文件")
        return

    print(f"找到{len(image_files)}个图像文件，开始处理...")

    for i, img_path in enumerate(image_files):
        print(f"处理第{i + 1}/{len(image_files)}个图像: {os.path.basename(img_path)}")
        file_name = os.path.splitext(os.path.basename(img_path))[0] + '_rgb.jpg'
        output_path = os.path.join(output_folder, file_name)

        process_sentinel2_image(img_path, output_path, bands, show_image, visualize, use_gpu)

    print("批量处理完成!")


# 使用示例
if __name__ == "__main__":
    # 单个图像处理示例 - 会显示原图和处理后的图像对比
    image_path = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    output_folder = "path_to_save_rgb_images/"

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.splitext(os.path.basename(image_path))[0] + '_rgb.jpg'
    output_path = os.path.join(output_folder, file_name)

    # 处理单个图像并显示对比
    rgb_data = process_sentinel2_image(image_path, output_path, bands=[4, 3, 2])

    # 批量处理示例 - 默认不显示图像，只保存
    folder_path = "path_to_sentinel2_images_folder"
    output_folder = "path_to_save_rgb_images_batch"
    bands = [4, 3, 2]  # 红、绿、蓝波段

    # 取消注释以下行以启用批量处理（默认不显示所有图像，只保存结果）
    # batch_process_sentinel2(folder_path, output_folder, bands)