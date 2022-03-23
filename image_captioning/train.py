import os
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from config import *
from dataset import CocoDataset
from model import CNN, LSTM
from utils import collate_function


if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    

DATASET = CocoDataset(
    root='datasets',
    download=True,
    transform=TRANSFORM,
    train=True
)

VOCABULARY = DATASET.vocabulary

DATA_LOADER = DataLoader(
    dataset=DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_function
)

encoder_ckpt = os.path.join(MODEL_SAVE_DIR, f'encoder-{4}-{MODEL_SAVE_PERIOD}.ckpt')
decoder_ckpt = os.path.join(MODEL_SAVE_DIR, f'decoder-{4}-{MODEL_SAVE_PERIOD}.ckpt')

ENCODER = CNN(EMBEDDING_SIZE).to(DEVICE)
DECODER = LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, len(VOCABULARY), 1).to(DEVICE)
parameters = list(DECODER.parameters()) + list(ENCODER.linear_layer.parameters()) + list(ENCODER.batchnorm.parameters())

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(parameters, lr=LEARNING_RATE)

def train(data_loader, encoder, decoder, criterion, optimizer, saved_encoder=None, saved_decoder=None):
    if os.path.exists(saved_encoder) and os.path.exists(saved_decoder):
        encoder.load_state_dict(torch.load(encoder_ckpt))
        decoder.load_state_dict(torch.load(decoder_ckpt))
        cur_epoch = int(encoder_ckpt.split('/')[-1].split('-')[1])
    else:
        cur_epoch = 0

    total_steps = len(data_loader)
    for epoch in range(cur_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        for i, (imgs, captions, caption_length) in enumerate(data_loader):
            imgs = imgs.to(DEVICE)
            captions = captions.to(DEVICE)
            targets = pack_padded_sequence(captions, caption_length, batch_first=True)[0]

            encoded = encoder(imgs)
            outputs = decoder(encoded, captions, caption_length)
            loss = criterion(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i%TRAIN_LOG_PERIOD == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i:4d}/{total_steps}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')

            if (i+1)%MODEL_SAVE_PERIOD == 0:
                torch.save(decoder.state_dict(), os.path.join(MODEL_SAVE_DIR, f'decoder-{epoch+1}-{i+1}.ckpt'))
                print(f'Saved decoder state_dict at {os.path.join(MODEL_SAVE_DIR, f"decoder-{epoch+1}-{i+1}.ckpt")}')
                torch.save(encoder.state_dict(), os.path.join(MODEL_SAVE_DIR, f'encoder-{epoch+1}-{i+1}.ckpt'))
                print(f'Saved encoder state_dict at {os.path.join(MODEL_SAVE_DIR, f"encoder-{epoch+1}-{i+1}.ckpt")}')

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f'took {epoch_time:4f}s/Epoch')
        
if __name__ == '__main__':
    train(DATA_LOADER, ENCODER, DECODER, CRITERION, OPTIMIZER, encoder_ckpt, decoder_ckpt)