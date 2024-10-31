import h5py
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio
import os
import numpy as np

def read_hdf5_and_save_as_tiff(hdf5_file_path, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开 HDF5 文件
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # 遍历 HDF5 文件中的所有数据集
        for dataset_name in hdf5_file.keys():
            # 读取图像数据
            dataset = hdf5_file[dataset_name][:]
            
            for (idx, image_data) in enumerate(dataset):
                if idx > 50: # 指定生成数据集大小
                    break

                # 检查图像数据的形状 (height, width) 或 (channels, height, width)
                if image_data.shape[0] > 1:
                    channels, height, width = image_data[[1,2,4,7], :].shape
                else:
                    channels, height, width = image_data.shape

                # 重新量化: 将像素值从 [0, 2047] 映射到 [0, 255]
                # 注意：假设11位图像的最大值是2047
                image_data = (image_data / 2047) * 255  # 归一化并缩放到[0, 255]
                image_data = np.clip(image_data, 0, 255)  # 确保所有值在 [0, 255] 范围内
                image_data = image_data.astype(np.uint8)  # 转换为8位整数
                
                # 设置 TIFF 文件的路径
                tiff_file_path = os.path.join(output_dir, dataset_name, f"{dataset_name}_{idx}.tif")

                # 保存为 TIFF 格式，确保 float 数据类型被支持
                with rasterio.open(
                    tiff_file_path, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=channels,
                    dtype=rasterio.uint8
                ) as dst:
                    for i in range(channels):
                        # 注意：rasterio 的通道索引从 1 开始
                        dst.write(image_data[i], i + 1)

                print(f"{dataset_name} 图像已保存为 {tiff_file_path}")


# 示例使用
hdf5_file_path = '/home/Shawalt/Demos/ImageFusion/DataSet/WorldView-3/valid_wv3.h5'  # 替换为你的 H5 文件路径
output_dir = '/home/Shawalt/Demos/ImageFusion/FAME-Net/wv3_data_dev/valid'  # 图像保存的文件夹

generate = True
if generate is True:
    read_hdf5_and_save_as_tiff(hdf5_file_path, output_dir)

# 生成图像DCT


# img = gdal.Open("/home/Shawalt/Demos/ImageFusion/FAME-Net/wv3_data/train/mask/mask_166.tif", gdal.GA_ReadOnly)
# band = img.GetRasterBand(1).ReadAsArray()  # 波段序号从1开始，而不是0
# plt.figure(figsize=(10, 10))
# plt.imshow(band, cmap='gray')
# plt.savefig('mask.png')

# img = gdal.Open("/home/Shawalt/Demos/ImageFusion/FAME-Net/wv3_data/train/gt/gt_166.tif", gdal.GA_ReadOnly)
# band = img.GetRasterBand(1).ReadAsArray()  # 波段序号从1开始，而不是0
# plt.figure(figsize=(10, 10))
# plt.imshow(band)
# plt.savefig('gt.png')

# red = img.GetRasterBand(1).ReadAsArray()
# green = img.GetRasterBand(4).ReadAsArray()
# blue = img.GetRasterBand(7).ReadAsArray()
#  # 将波段数据组合成 RGB 图像
# rgb_image = np.dstack((red, green, blue))

# # 归一化数据以适合显示
# rgb_image = np.clip(rgb_image, 0, 255)  # 确保像素值在 [0, 255] 范围内
# rgb_image = rgb_image.astype(np.uint8)  # 转换为无符号8位整型

# # 显示伪彩色图像
# plt.figure(figsize=(10, 10))
# plt.imshow(rgb_image)
# plt.savefig('pseduo.png')
