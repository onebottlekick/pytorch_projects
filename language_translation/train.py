import time
import math

import torch
import torch.nn as nn

from configs import *
from models import Encoder, Decoder, Seq2Seq
from utils import epoch_time, init_weights
from dataset import SRC, TRG, train_iterator, val_iterator


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg        
        optimizer.zero_grad()
        
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss/len(iterator)


if __name__ == '__main__':
    encoder = Encoder(len(SRC.vocab), len(TRG.vocab), HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(len(TRG.vocab), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters())
    target_pad_idx = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)
        
    best_val_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        val_loss = evaluate(model, val_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f'Epoch {epoch+1:02} | Time {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'Val Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}')