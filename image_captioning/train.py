import argparse
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


def train(data_loader, encoder, decoder, criterion, optimizer, saved_encoder=None, saved_decoder=None):
    if saved_encoder and saved_decoder:
        encoder.load_state_dict(torch.load(encoder_ckpt))
        decoder.load_state_dict(torch.load(decoder_ckpt))
        cur_epoch = int(encoder_ckpt.split('/')[-1].split('-')[1])
    else:
        cur_epoch = 0

    total_steps = len(data_loader)
    for epoch in range(cur_epoch, args.num_epochs):
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

            if i%args.train_log_period == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i:4d}/{total_steps}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')

            if (i+1)%args.model_save_period == 0:
                torch.save(decoder.state_dict(), os.path.join(args.model_save_dir, f'decoder-{epoch+1}-{i+1}.ckpt'))
                print(f'Saved decoder state_dict at {os.path.join(args.model_save_dir, f"decoder-{epoch+1}-{i+1}.ckpt")}')
                torch.save(encoder.state_dict(), os.path.join(args.model_save_dir, f'encoder-{epoch+1}-{i+1}.ckpt'))
                print(f'Saved encoder state_dict at {os.path.join(args.model_save_dir, f"encoder-{epoch+1}-{i+1}.ckpt")}')

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f'took {epoch_time:4f}s/Epoch')
        
if __name__ == '__main__':
    epilog = 'example: python train.py --batch_size 256 --num_epochs 5 --load_model 2'
    parser = argparse.ArgumentParser(description='Train Image Captioning Model - Encoder(CNN)->Decoder(LSTM). \
        Default arguments are in config.py', epilog=epilog)
    
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size of dataset')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='number of train epochs')
    parser.add_argument('--learning_rate', '--lr', type=float, default=LEARNING_RATE, help='learning rate')
    parser.add_argument('--device', type=str, default=DEVICE, help='device to train model')
    
    parser.add_argument('--embedding_size', type=int, default=EMBEDDING_SIZE, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE, help='hidden size')
    
    parser.add_argument('--load_model', default=None, help='index of saved model')
    parser.add_argument('--model_save_dir', type=str, default=MODEL_SAVE_DIR, help='path to save the model')
    parser.add_argument('--model_save_period', type=int, default=MODEL_SAVE_PERIOD, help='iterations to save the model')
    parser.add_argument('--train_log_period', type=int, default=TRAIN_LOG_PERIOD, help='train log interval(iterations)')
    args = parser.parse_args()
    
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)


    DATASET = CocoDataset(
        root='datasets',
        download=True,
        transform=TRANSFORM,
        train=True
    )

    VOCABULARY = DATASET.vocabulary

    DATA_LOADER = DataLoader(
        dataset=DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_function
    )
    
    ENCODER = CNN(args.embedding_size).to(args.device)
    DECODER = LSTM(args.embedding_size, args.hidden_size, len(VOCABULARY), 1).to(args.device)
    parameters = list(DECODER.parameters()) + list(ENCODER.linear_layer.parameters()) + list(ENCODER.batchnorm.parameters())

    CRITERION = nn.CrossEntropyLoss()
    OPTIMIZER = optim.Adam(parameters, lr=args.learning_rate)
    
    if args.load_model:
        decoder_ckpt = os.path.join(args.model_save_dir, f'decoder-{args.load_model}-{args.model_save_period}.ckpt')
        encoder_ckpt = os.path.join(args.model_save_dir, f'encoder-{args.load_model}-{args.model_save_period}.ckpt')
        train(DATA_LOADER, ENCODER, DECODER, CRITERION, OPTIMIZER, encoder_ckpt, decoder_ckpt)
    else:
        train(DATA_LOADER, ENCODER, DECODER, CRITERION, OPTIMIZER)