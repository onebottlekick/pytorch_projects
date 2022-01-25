import os
import tarfile

from PIL import Image
from urllib import request

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n)


class VOC2012(Dataset):
    def __init__(self, root='../datasets', download=False, train=True, transform=None):
        self.root = root
        if download:
            self.download_extract_VOC()
        else:
            print('Already Exists!')
        self.train = train
        try:
            self.img_list, self.annotation_list = self.make_data_path_list()
        except Exception as e:
            print(f'{e}\nYou need to change download=True')
        self.transform = transform
    
    def __len__(self):
        assert len(self.img_list) == len(self.annotation_list)
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])        
        annotation_class_img = Image.open(self.annotation_list[index])
        
        if self.transform:
            img = self.transform[0](img)
            annotation_class_img = self.transform[1](annotation_class_img)
        else:
            img, annotation_class_img = transforms.ToTensor()(img), transforms.ToTensor()(annotation_class_img)
        
        return img, annotation_class_img
    
    def download_extract_VOC(self):
        os.makedirs(self.root, exist_ok=True)
        
        URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        SAVE_PATH = os.path.join(self.root, 'VOCdataset.tar')
    
        if not os.path.exists(SAVE_PATH):
            with DownloadProgress(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc='VOCdataset.tar') as t:
                request.urlretrieve(URL, SAVE_PATH, reporthook=t.update_to)
    
            with tarfile.TarFile(SAVE_PATH) as f:
                f.extractall(self.root)
    
    def make_data_path_list(self):
        data_path = os.path.join(self.root, 'VOCdevkit', 'VOC2012')
        
        if self.train:
            id_names = os.path.join(data_path, 'ImageSets', 'Segmentation', 'train.txt')
        else:
            id_names = os.path.join(data_path, 'ImageSets', 'Segmentation', 'val.txt')
            
        img_list = []
        annotation_list = []
        for line in open(id_names):
            file_id = line.strip()

            img_path = os.path.join(data_path, 'JPEGImages', f'{file_id}.jpg')
            img_list.append(img_path)            

            annotation_path = os.path.join(data_path, 'SegmentationClass', f'{file_id}.png')
            annotation_list.append(annotation_path)
                
        return img_list, annotation_list