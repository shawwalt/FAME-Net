#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-01-19 20:57:29
@Description: file content
'''
import re
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from scipy.io import loadmat
from random import randrange
import torch.nn.functional as F
import torchvision.transforms.functional as F
from torchvision import transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF', '.mat'])


def load_img(filepath, type='Tensor'):
    img_data = None
    img = None

    if filepath.split('.')[-1] == 'mat':
        mat_data = loadmat(filepath)
        try:
            img_data = mat_data['imgMS']
        except KeyError:
            img_data = mat_data['imgPAN']
            img_data = img_data[:, :, np.newaxis]
        if type == 'PIL':
            img_data = img_data / 2047 * 255
            img = Image.fromarray(img_data.astype('uint8'))
        else:
            img_data = img_data / 2047 * 255 # 先除后乘，防止高位溢出
            img = torch.from_numpy(img_data.astype('uint8'))
            img = img.permute(2, 0, 1)
    else:
        img = Image.open(filepath)
    #img = Image.open(filepath)

    return img

def rescale_img(img_in, scale):
    if isinstance(img_in, Image.Image):
        size_in = img_in.size
        new_size_in = tuple([int(x * scale) for x in size_in])
        img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
        return img_in
    elif isinstance(img_in, torch.Tensor):
        size_in = img_in.shape
        new_h, new_w = tuple([int(size_in[i] * scale) for i in range(1, 3)])
        img_in = np.resize(img_in, (size_in[0], new_h, new_w))
        return img_in


def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    if isinstance(ms_image, Image.Image):
        (ih, iw) = lms_image.size
        (th, tw) = (scale * ih, scale * iw)

        patch_mult = scale #if len(scale) > 1 else 1
        tp = patch_mult * patch_size
        ip = tp // scale

        if ix == -1:
            ix = random.randrange(0, iw - ip + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip + 1)

        (tx, ty) = (scale * ix, scale * iy)

        lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
        ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
        pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
        bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
    elif isinstance(ms_image, torch.Tensor):
        _, ih, iw = lms_image.size()
        (th, tw) = (scale * ih, scale * iw)

        patch_mult = scale #if len(scale) > 1 else 1
        tp = patch_mult * patch_size
        ip = tp // scale

        if ix == -1:
            ix = random.randrange(0, iw - ip + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip + 1)
        (tx, ty) = (scale * ix, scale * iy)

        lms_image = F.crop(lms_image, iy, ix, ip, ip)
        ms_image = F.crop(ms_image, ty, tx, tp, tp)
        pan_image = F.crop(pan_image, ty, tx, tp, tp)
        bms_image = F.crop(bms_image, ty, tx, tp, tp)
    
    info_patch = {
            'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch

def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        if isinstance(ms_image, Image.Image):
            ms_image = ImageOps.flip(ms_image)
            lms_image = ImageOps.flip(lms_image)
            pan_image = ImageOps.flip(pan_image)
            bms_image = ImageOps.flip(bms_image)
        elif isinstance(ms_image, torch.Tensor):
            flip_transform = transforms.RandomHorizontalFlip(p=1.0)
            ms_image = flip_transform(ms_image)
            lms_image = flip_transform(lms_image)
            pan_image = flip_transform(pan_image)
            bms_image = flip_transform(bms_image)
        info_aug['flip_h'] = True 

    if rot:
        if random.random() < 0.5:
            if isinstance(ms_image, Image.Image):
                ms_image = ImageOps.mirror(ms_image)
                lms_image = ImageOps.mirror(lms_image)
                pan_image = ImageOps.mirror(pan_image)
                bms_image = ImageOps.mirror(bms_image)
            elif isinstance(ms_image, torch.Tensor):
                flip_transform = transforms.RandomHorizontalFlip(p=1.0)
                ms_image = flip_transform(ms_image)
                lms_image = flip_transform(lms_image)
                pan_image = flip_transform(pan_image)
                bms_image = flip_transform(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            if isinstance(ms_image, Image.Image):
                ms_image = ms_image.rotate(180)
                lms_image = lms_image.rotate(180)
                pan_image = pan_image.rotate(180)
                bms_image = pan_image.rotate(180)
            elif isinstance(ms_image, torch.Tensor):
                flip_transform = transforms.RandomRotation(degrees=180)
                ms_image = flip_transform(ms_image)
                lms_image = flip_transform(lms_image)
                pan_image = flip_transform(pan_image)
                bms_image = flip_transform(bms_image)
            info_aug['trans'] = True
            
    return ms_image, lms_image, pan_image, bms_image, info_aug

def sort_key(x):
    file_name = x.split('/')[-1]
    return int(file_name.split('.')[0])

class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None, data_dir_mask=None):
        super(Data, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in sorted(listdir(data_dir_ms), key=sort_key) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in sorted(listdir(data_dir_pan), key=sort_key) if is_image_file(x)]
        self.mask_image_filenames = None
        if os.path.isdir(data_dir_mask):
            self.mask_image_filenames = [join(data_dir_mask, x) for x in sorted(listdir(data_dir_mask), key=sort_key) if is_image_file(x)]
        data_dir_mask = os.path.join(*data_dir_pan.split("\\")[:-1],"mask")
        self.mask_filenames = [join(data_dir_pan, x) for x in sorted(listdir(data_dir_pan), key=sort_key) if is_image_file(x)]
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        lms_image = None
        if self.mask_image_filenames != None:
            mask_image = load_img(self.mask_image_filenames[index])
            mask_image = mask_image.crop((0, 0, mask_image.size[0] // self.upscale_factor * self.upscale_factor,
                                          mask_image.size[1] // self.upscale_factor * self.upscale_factor))
            mask_image = mask_image.convert('L')
            mask_image = mask_image.resize((int(mask_image.size[0] / self.upscale_factor), int(mask_image.size[1] / self.upscale_factor))
                                           , Image.NEAREST)
            mask_image = mask_image.point(lambda x: 255 if x > 0 else 0)
            # mask_image = torch.tensor(np.array(mask_image))
        _, file = os.path.split(self.ms_image_filenames[index])
        
        if isinstance(ms_image, Image.Image):
            ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                      ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        elif isinstance(ms_image, torch.Tensor):
            ms_image = F.crop(ms_image, 0, 0, ms_image.size()[1] // self.upscale_factor * self.upscale_factor, 
                              ms_image.size()[2] // self.upscale_factor * self.upscale_factor)
        if isinstance(ms_image, Image.Image):
            lms_image = ms_image.resize(
                (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        elif isinstance(ms_image, torch.Tensor):
            resize_transform = transforms.Resize(
                (int(ms_image.size()[1] / self.upscale_factor), int(ms_image.size()[2] / self.upscale_factor)),
                interpolation=transforms.InterpolationMode.BICUBIC
            ) 
            lms_image = resize_transform(ms_image)
        if isinstance(pan_image, Image.Image):
            pan_image = pan_image.resize(
                (int(pan_image.size[0] / self.upscale_factor), int(pan_image.size[1] / self.upscale_factor)), Image.BICUBIC)
            pan_image = pan_image.convert('L')
            pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                        pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        elif isinstance(pan_image, torch.Tensor):
            resize_transform = transforms.Resize(
                (int(pan_image.size()[1] / self.upscale_factor), int(pan_image.size()[2] / self.upscale_factor)),
                interpolation=transforms.InterpolationMode.BICUBIC
            ) 
            pan_image = resize_transform(pan_image)
            pan_image = F.crop(pan_image, 0, 0, pan_image.size()[1] // self.upscale_factor * self.upscale_factor, 
                                         pan_image.size()[2] // self.upscale_factor * self.upscale_factor)
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.mask_image_filenames:
            ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, mask_image,
                                                                     self.patch_size, scale=self.upscale_factor)
        else:
            ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image,
                                                                     self.patch_size, scale=self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform and isinstance(ms_image, Image.Image):
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)
        else:
            ms_image = ms_image / 255.0
            lms_image = lms_image / 255.0
            pan_image = pan_image / 255.0
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
            
        if self.mask_image_filenames:
            hf = bms_image
            lf = 1 - hf
            bms_image = torch.cat([hf, lf], dim=0)
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None,data_dir_mask=None):
        super(Data_test, self).__init__()
        print(data_dir_mask)
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index]) #128
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor), int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            #print(ms_image.max())
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)
            #print(torch.max(ms_image))
            #print(torch.min(ms_image))

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

 # transfer ms instead of lms when no-ref
        # lms_image = ms_image
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class Data_eval(data.Dataset):
    def __init__(self, image_dir, upscale_factor, cfg, transform=None):
        super(Data_eval, self).__init__()
        
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        lms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        lms_image = ms_image.crop((0, 0, lms_image.size[0] // self.upscale_factor * self.upscale_factor, lms_image.size[1] // self.upscale_factor * self.upscale_factor))
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
        
        if self.data_augmentation:
            lms_image, pan_image, bms_image, _ = augment(lms_image, pan_image, bms_image)
        
        if self.transform:
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
            
        return lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)