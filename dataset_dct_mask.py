import re
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import h5py
import os
import rasterio

def create_circular_mask(h, w, center=None, radius=None):
    """
    生成一个圆形掩码，圆内的区域为1，其他区域为0。
    
    参数:
    h: 掩码的高度
    w: 掩码的宽度
    center: 圆心坐标 (默认为图像中心)
    radius: 圆的半径 (默认为最小的边的一半)

    返回:
    一个形状为 (h, w) 的二维掩码，圆内的值为 1，圆外的值为 0。
    """
    if center is None:
        center = (int(w / 2), int(h / 2))  # 默认圆心为图像中心
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])  # 默认半径为图像最小边长的一半

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # 圆内区域为1，圆外区域为0
    mask = dist_from_center <= radius
    return mask


def apply_dct_on_bands(tiff_file_path, low_frequency_radius):
    # 打开 TIFF 文件
    dataset = gdal.Open(tiff_file_path)

    if dataset is None:
        print(f"无法打开文件: {tiff_file_path}")
        return

    # 获取波段数量
    num_bands = dataset.RasterCount
    print(f"图像共有 {num_bands} 个波段")

    band_data = dataset.GetRasterBand(1).ReadAsArray()
    h, w = band_data.shape
    idct_ms = np.zeros([num_bands, h, w])
    for band_idx in range(1, num_bands + 1):
        # 读取每个波段的图像数据
        band_data = dataset.GetRasterBand(band_idx).ReadAsArray()

        # 对该波段进行二维DCT变换
        dct_band = dct(dct(band_data.T, norm='ortho').T, norm='ortho')

        # 生成掩码滤除低频信息
        rows, cols = dct_band.shape
        mask = np.ones((rows, cols))
        
        h, w = band_data.shape
        mask = create_circular_mask(h, w, center=(0, 0), radius=low_frequency_radius)
        
        # 应用掩码滤除低频
        dct_band_filtered = dct_band * (1 - mask)

        # 对滤波后的图像进行逆DCT变换
        idct_band = idct(idct(dct_band_filtered.T, norm='ortho').T, norm='ortho')
        idct_ms[band_idx-1] = idct_band
    return idct_ms


def generate_frequency_mask(ms):
    if(len(ms.shape) != 3):
        print("输入需要为ms图像")
        return
    
    num_bands, h, w = ms.shape
    frequency_masks = np.zeros([num_bands, h, w])
    for (band_idx, band_data) in enumerate(ms):
        # 归一化波段到 [0, 1] 范围
        min_val, max_val = np.min(band_data), np.max(band_data)
        if max_val > min_val:
            normalized_band = (band_data - min_val) / (max_val - min_val)
        else:
            normalized_band = np.zeros_like(band_data)  # 避免除零错误
        
        # 计算前40%的阈值
        threshold_value = np.percentile(normalized_band, 80)

        # 阈值化处理：小于等于阈值的置为1，其余置为0
        binary_band = (normalized_band >= threshold_value).astype(np.uint8) * 255
        frequency_masks[band_idx] = binary_band
    return frequency_masks

# 示例使用
output_dir = '/home/Shawalt/Demos/ImageFusion/FAME-Net_back/FAME-Net/'  # 图像保存的文件夹

tiff_file_path = '/home/Shawalt/Demos/ImageFusion/DataSet/NBU_DataSet/Satellite_Dataset/DataSet_TIF/6_WorldView-3/PAN_1024'  # 替换为实际的 TIFF 文件路径

 # 遍历目录及其子目录中的所有文件

tif_files = []
for root, dirs, files in os.walk(tiff_file_path):
    for file in files:
        if file.lower().endswith('.tif'):
            # 生成完整路径并加入列表
            file_path = os.path.join(root, file)
            tif_files.append(file_path)

def sort_key(x):
    file_name = x.split('/')[-1]
    return int(file_name.split('.')[0])

tif_files = sorted(tif_files, key=sort_key)
for idx, img_path in enumerate(tif_files):
    idct_ms = apply_dct_on_bands(img_path, low_frequency_radius=15)
    frequency_masks = generate_frequency_mask(idct_ms)
    c, h, w = frequency_masks.shape
    output_file_path = os.path.join(output_dir, 'mask', f'{idx+1}.tif')
    with rasterio.open(
            output_file_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=c,
            dtype=np.uint8
        ) as dst:
            for i in range(c):
                dst.write(frequency_masks[i], i + 1)
    print(f'已写入{os.path.abspath(output_file_path)}')

