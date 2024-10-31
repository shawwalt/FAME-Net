from PIL import Image, ImageOps
import os

def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    return img
# 使用NEAREST比双线性插值效果好
idx = 1
img_path = os.path.join('.', 'mask', f'{idx}.tif')
mask_image = load_img(img_path)
mask_image = mask_image.resize((int(mask_image.size[0] / 4), int(mask_image.size[1] / 4))
                                , Image.NEAREST)
mask_image = mask_image.point(lambda x: 255 if x > 0 else 0)
mask_image.save(f'{idx}_down_4.tif', format='TIFF')