import os

import torch
from torchvision import transforms

from config import *
from dataset import CocoDataset
from model import CNN, LSTM
from utils import captioning, load_test_img, plot_captioned_img


epoch = 5
encoder_ckpt = os.path.join(MODEL_SAVE_DIR, f'encoder-{epoch}-{MODEL_SAVE_PERIOD}.ckpt')
decoder_ckpt = os.path.join(MODEL_SAVE_DIR, f'decoder-{epoch}-{MODEL_SAVE_PERIOD}.ckpt')
img_path = 'sample.jpg'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

data = CocoDataset()
vocabulary = data.vocabulary

encoder = CNN(EMBEDDING_SIZE).eval().to(DEVICE)
decoder = LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, len(vocabulary), 1).to(DEVICE)

encoder.load_state_dict(torch.load(encoder_ckpt))
decoder.load_state_dict(torch.load(decoder_ckpt))

img = load_test_img(img_path, transform)
img = img.to(DEVICE)

feature = encoder(img)
sampled_idx = decoder.sample(feature).squeeze(0).cpu().numpy()

caption = captioning(sampled_idx, vocabulary)

plot_captioned_img(img_path, caption, epoch, save=True, show=False)