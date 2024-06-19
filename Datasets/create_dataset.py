'''
Split dataset as train, test, val  6:2:2
use function dataset_wrap, return {train:, val:, test:} torch dataset

datasets names: isic2018, PH2, DMF, SKD
'''

import os
import json
import torch
import random
import numpy as np
from torchvision import transforms
import albumentations as A
import pandas as pd
from Datasets.transform import *

dataset_indices = {
    'isic2018': 0,
    'PH2': 1,
    'DMF': 2,
    'SKD': 3,
}

def norm01(x):
    return np.clip(x, 0, 255) / 255


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        
        super(SkinDataset, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        p = 0.5
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            A.RandomBrightnessContrast(p=p),
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5

        if self.use_aug:
            tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        else:
            tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        img_data = norm01(img_data)
        label_data = np.expand_dims(label_data, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        img_data = self.normalize(img_data)


        return{
            'image': img_data,
            'label': label_data,
            'name': sample_name.replace('.npy', ''),
        }


    def __len__(self):
        return self.num_samples


class StrongWeakAugment(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        
        super(StrongWeakAugment, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        w_p = 0.5
        s_p = 1.0
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(p=w_p),
            A.HorizontalFlip(p=w_p),
            A.VerticalFlip(p=w_p),
            A.ShiftScaleRotate(p=w_p),
            A.RandomBrightnessContrast(p=w_p),
        ])
        self.strong_augment = A.Compose([
            A.GaussNoise(p=s_p),
            A.ColorJitter(p=s_p),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)

        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']
        
        
        img_w = norm01(img_w)
        img_s = norm01(img_s)

        img_s = torch.from_numpy(img_s).float()
        img_w = torch.from_numpy(img_w).float()
        
        img_s = img_s.permute(2, 0, 1)
        img_w = img_w.permute(2, 0, 1)
        
        img_s = self.normalize(img_s)
        img_w = self.normalize(img_w)

        return{
            'img_w': img_w,
            'img_s': img_s,
        }


    def __len__(self):
        return self.num_samples
    
class StrongWeakAugment2(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(StrongWeakAugment2, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        w_p = 0.5
        s_p = 1.0
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=w_p),
            A.VerticalFlip(p=w_p)
        ])
        self.strong_augment = A.Compose([
            A.GaussNoise(p=s_p),
            A.RandomBrightnessContrast(p=s_p),
            A.ColorJitter(p=s_p)
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)

        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']
        
        img_w = norm01(img_w)
        img_s = norm01(img_s)
       
        img_w = torch.from_numpy(img_w).float()
        img_s = torch.from_numpy(img_s).float()

        img_w = img_w.permute(2, 0, 1)
        img_s = img_s.permute(2, 0, 1)
        
        img_w = self.normalize(img_w)
        img_s = self.normalize(img_s)

        return{
            'id': index,
            'img_w': img_w,
            'img_s': img_s,
        }


    def __len__(self):
        return self.num_samples

class SkinDataset2(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(SkinDataset2, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        p = 0.5
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5

        if self.use_aug:
            tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        else:
            tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        img_data = norm01(img_data)
        label_data = np.expand_dims(label_data, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        img_data = self.normalize(img_data)


        return{
            'name': sample_name,
            'image': img_data,
            'label': label_data,
        }


    def __len__(self):
        return self.num_samples



def get_dataset(args, img_size=384, supervised_ratio=0.2, train_aug=False, k=6, lb_dataset=SkinDataset, ulb_dataset=SkinDataset, v_dataset=SkinDataset):
    
    folds = []
    for idx in range(1, 6):
        fold = []
        with open(f'proceeded_data/fold{idx}.txt', 'r') as f:
            fold = [line.replace('\n', '') for line in f.readlines()]
            folds.append(fold)
    
    train_data = []
    for j in range(5):
        if j != k - 1:
            train_data = [*train_data, *folds[j]]
    train_data = sorted(train_data)
    l_data = sorted(random.sample(train_data, int(len(train_data) * supervised_ratio)))
    u_data = sorted([sample for sample in train_data if sample not in l_data])
    l_dataset = lb_dataset(dataset=l_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
    u_dataset = ulb_dataset(dataset=u_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
        
    val_data = sorted(folds[k - 1])
    val_dataset = v_dataset(dataset=val_data, img_size=img_size, use_aug=False, data_path=args.data.val_folder)
        
    dataset = {
        'lb_dataset': l_dataset,
        'ulb_dataset': u_dataset,
        'val_dataset': val_dataset
    }
             
    return dataset
