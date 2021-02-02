import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

class DewDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.filepathes = df['filepath'].values
        self.label = df['label'].values
        self.gt = df['GT'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.filepathes[index]))
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)

        if self.transform is not None:
            rgbimg = self.transform(rgbimg)

        return rgbimg, self.label[index], self.gt[index]

    def __len__(self):
        return self.label.shape[0]


class DewCoralDataset(DewDataset):
    def __init__(self, csv_path, img_dir, transform=None, class_num=70, reverse=False):
        super().__init__(csv_path, img_dir, transform)
        self.reverse = reverse
        self.class_num = class_num

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.filepathes[index]))
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)

        if self.reverse:
            levels = [0]*self.label[index] + [1]*(self.class_num - 1 - self.label[index])
            levels = torch.tensor(levels, dtype=torch.float32)
        else:
            levels = [1]*self.label[index] + [0]*(self.class_num - 1 - self.label[index])
            levels = torch.tensor(levels, dtype=torch.float32)

        if self.transform is not None:
            rgbimg = self.transform(rgbimg)

        return rgbimg, self.label[index], levels, self.gt[index]


class DewCoralObjectDataset(DewDataset):
    def __init__(self, csv_path, img_dir, transform=None, class_num=70, reverse=False):
        super().__init__(csv_path, img_dir, transform)
        df = pd.read_csv(csv_path)
        self.left = df['left'].values
        self.top = df['top'].values
        self.right = df['right'].values
        self.bottom = df['bottom'].values
        self.reverse = reverse
        self.class_num = class_num

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.filepathes[index]))
        point = list(map(lambda x: round(float(x)), [self.left[index], self.top[index], self.right[index], self.bottom[index]]))
        cropped_img = img.crop(point)
        rgbimg = Image.new("RGB", cropped_img.size)
        rgbimg.paste(cropped_img)

        if self.reverse:
            levels = [0]*self.label[index] + [1]*(self.class_num - 1 - self.label[index])
            levels = torch.tensor(levels, dtype=torch.float32)
        else:
            levels = [1]*self.label[index] + [0]*(self.class_num - 1 - self.label[index])
            levels = torch.tensor(levels, dtype=torch.float32)

        if self.transform is not None:
            rgbimg = self.transform(rgbimg)

        return rgbimg, self.label[index], levels, self.gt[index]
