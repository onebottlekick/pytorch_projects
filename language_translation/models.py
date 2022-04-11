import random

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)

    def forward(self, source):
        embedded = self.dropout(self.embedding(source))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(outputs.squeeze(0))
        return prediction, hidden, cell
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        target_length = target.shape[0]
        target_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(source)
        input = target[0, :]
        
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top = output.argmax(1)
            input = target[t] if teacher_force else top
        return outputs