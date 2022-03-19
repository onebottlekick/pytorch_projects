import os
import pickle
import urllib
import zipfile
from PIL import Image
from collections import Counter

import nltk
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import DownloadProgressBar


class Vocab:
    def __init__(self, root, json, threshold, save_vocab=True):
        self.root = root
        self.json = json
        self.threshold = threshold
        
        self.w2i = {}
        self.i2w = {}
        self.index = 0
        
        nltk.download('punkt')
        
        if not os.path.exists(os.path.join(self.root, 'vocabulary.pkl')):        
            self.build_vocabulary()
            if save_vocab:
                self.save_vocab()
            with open(os.path.join(self.root, 'vocabulary.pkl'), 'rb') as f:
                self.vocabulary = pickle.load(f)
        else:
            print('Vocabulary already exists!')
            with open(os.path.join(self.root, 'vocabulary.pkl'), 'rb') as f:
                self.vocabulary = pickle.load(f)
        
    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
    
    def __len__(self):
        return len(self.w2i)
    
    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1
            
    def build_vocabulary(self):
        coco = COCO(self.json)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            
            if (i+1)%1000 == 0:
                print(f'[{i+1}/{len(ids)}] Tokenized the captions.')
        
        tokens = [token for token, cnt in counter.items() if cnt >= self.threshold]
        
        self.add_token('<pad>')
        self.add_token('<start>')
        self.add_token('<end>')
        self.add_token('<unk>')

        for token in tokens:
            self.add_token(token)
            
    def save_vocab(self):
        with open(os.path.join(self.root, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(self, f)
        print(f'Total vocabulary size: {len(self)}')
        print(f"Saved the vocabulary wrapper to {os.path.join(self.root, 'vocabulary.pkl')}")
        
        
class CocoDataset(Dataset):
    def __init__(self, root='datasets', download=False, transform=None, train=True):
        self.root = root
        self.train = train   
        if download and not os.path.exists(self.root):
            self.download_data("http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip")
            self.download_data("http://images.cocodataset.org/zips/train2014.zip")
            self.download_data("http://images.cocodataset.org/zips/val2014.zip")
        else:
            print('Dataset already exists!')
        
        
        self.coco_data = COCO(os.path.join(self.root, 'annotations', 'captions_train2014.json'))
        self.indices = list(self.coco_data.anns.keys())
        self.transform = transform
            
        vocab = Vocab(self.root, os.path.join(self.root, 'annotations', 'captions_train2014.json'), 4)
        self.vocabulary = vocab.vocabulary
    
    def download_data(self, url):
        print(f'Downloading {url}...')
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            zip_path, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as f:
            for name in tqdm(iterable=f.namelist(), total=len(f.namelist())):
                f.extract(member=name, path=self.root)
                
    def __getitem__(self, idx):
        coco_data = self.coco_data
        vocabulary = self.vocabulary
        annotation_id = self.indices[idx]
        caption = coco_data.anns[annotation_id]['caption']
        img_id = coco_data.anns[annotation_id]['image_id']
        img_path = coco_data.loadImgs(img_id)[0]['file_name']
        
        if self.train:
            img = Image.open(os.path.join(self.root, 'train2014', img_path)).convert('RGB')
        else:
            img = Image.open(os.path.join(self.root, 'val2014', img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocabulary('<start>'))
        caption.extend([vocabulary(token) for token in word_tokens])
        caption.append(vocabulary('<end>'))
        ground_truth = torch.tensor(caption)
        return img, ground_truth
    
    def __len__(self):
        return len(self.indices)