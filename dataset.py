from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import math
import random
import torch
import numpy as np
import os

class LNENDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.list_file = args.list_dir
        self.evaluate = args.evaluate
        # load dataset
        self.x = self.load_dataset_folder()
        # set transforms
        if not self.evaluate:
            self.transform = Transform()
        else:
            self.transform = Transform_Evaluation()

    def __getitem__(self,  idx):
        paths = self.x[idx]
        x = Image.open(paths)
        x1, x2 = self.transform(x)
        if not self.evaluate:
            return x1, x2
        else:
            return x1, x2, paths


    def __len__(self):
        return len(self.x)
    
    def load_dataset_folder(self):
        list_file =  self.list_file
        x = []
        img_dir = os.path.join(list_file)
        with open(img_dir, 'r') as f:
            content =  f.readlines()
        files_list = []
        for l in content:
            l =  l.strip()
            files_list.append(l)
        files_list = sorted(files_list)
        x.extend(files_list)
        return list(x)
    
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.5)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2), # like in JLQ
            GaussianBlur(p=0.0), # False like in JLQ
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.5)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    
class Transform_Evaluation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(384, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return y1, y2

