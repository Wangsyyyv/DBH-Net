import torch.utils.data as data
import PIL.Image as Image
import os
import torchvision.transforms as transforms
import albumentations as A
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch


def make_dataset(train_root, mask_root):
    imgs = []
    train_filenames = os.listdir(train_root)
    labels = os.listdir(mask_root)
    for i in range(len(train_filenames)):
        img = os.path.join(train_root, train_filenames[i])
        mask = os.path.join(mask_root, labels[i])
        imgs.append((img, mask))
    return imgs


def standardization(img):
    # 将数据转为C,W,H，并归一化到[0，1]
    img = transforms.ToTensor()(img)
    # 需要对数据进行扩维，增加batch维度
    data = torch.unsqueeze(img, 0)
    nb_samples = 0.
    # 创建3维的空列表
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    N, C, H, W = data.shape[:4]
    data = data.view(N, C, -1)  # 将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
    # 展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
    channel_mean += data.mean(2).sum(0)
    # 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
    channel_std += data.std(2).sum(0)
    # 获取所有batch的数据，这里为1
    nb_samples += N
    # 获取同一batch的均值和标准差
    channel_mean /= nb_samples
    channel_std /= nb_samples
    trans = transforms.Normalize(channel_mean, channel_std)  # ->[-1,1]
    return trans(img)


class LiverDataset(data.Dataset):
    def __init__(self, root, label, train=True, transform=True):
        imgs = make_dataset(root, label)
        self.imgs = imgs
        self.transform = transform
        self.train = train
        self.tsfm = self.get_transform()
        self.target_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=cv2.INTER_NEAREST),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Resize((224, 224), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')
        if self.train:
            if random.random() < 0.5:
                if self.transform:
                    img_x = np.asarray(img_x)
                    img_y = np.asarray(img_y)
                    image = self.tsfm(image=img_x, mask=img_y)
                    img_x = image['image']
                    img_y = image['mask']
                    img_x = Image.fromarray(img_x)
                    img_y = Image.fromarray(img_y)
        img_x = self.image_transform(img_x)
        img_x = standardization(img_x)
        img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

    # 数据增强
    def get_transform(self):
        transform = A.Compose([
            # 随机旋转
            A.Rotate(),
            # Flip 垂直或水平和垂直翻转
            A.Flip(always_apply=False, p=0.5),
            A.OneOf([
                # 参数：随机色调、饱和度、值变化。
                A.HueSaturationValue(),
                # 亮度和饱和度
                A.RandomBrightnessContrast(),
            ], p=1),
            # ShiftScaleRotate 随机应用仿射变换：平移，缩放和旋转
            A.ShiftScaleRotate(),
            # ElasticTransform 弹性变换
            A.ElasticTransform(),
        ])

        return transform


def visual_img(imgs):
    for i, img in enumerate(imgs):
        img = img.numpy().transpose(1, 2, 0)
        print(img.shape)
        c_dim = img.shape[-1]
        if (c_dim == 1):
            img = img.reshape(img.shape[0:2])
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    root = "./data_gastric/train/images"
    label = "./data_gastric/train/masks"
    liver_dataset = LiverDataset(root, label, train=True, transform=True)
    dataloaders = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=0)

    for img_x, img_y in dataloaders:
        visual_img(img_x)
        visual_img(img_y)
        break
