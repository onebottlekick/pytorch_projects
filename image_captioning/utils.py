import os
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import IMG_SIZE


def collate_function(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs, 0)
    caption_length = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(caption_length)).long()
    for i, caption in enumerate(captions):
        end = caption_length[i]
        targets[i, :end] = caption[:end]
    return imgs, targets, caption_length


def captioning(sampled_idx, vocabulary):
    predicted_caption = []
    for token_idx in sampled_idx:
        word = vocabulary.i2w[token_idx]
        predicted_caption.append(word)
        if word == '<end>':
            break
    predicted = ' '.join(predicted_caption).lstrip('<start>').rstrip('<end>')
    return predicted


def load_test_img(img_path, transform=None):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    
    if transform:
        img = transform(img).unsqueeze(0)
    
    return img


def plot_captioned_img(img_path, caption, epoch, show=True, save=False, save_path='results', figname=None):
    img = Image.open(img_path)
    plt.title(f'Predicted Caption: {caption}')
    plt.imshow(np.asarray(img))
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, figname if figname else f'{img_path.split("/")[-1].split(".")[0]}_epoch{epoch}.png'))
    if show:
        plt.show()    


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n)