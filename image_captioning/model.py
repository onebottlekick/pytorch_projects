import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models


class CNN(nn.Module):
    def __init__(self, embedding_size):
        super(CNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        module_list = list(resnet.children())[:-1]
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batchnorm = nn.BatchNorm1d(embedding_size, momentum=0.01)
        
    def forward(self, img):
        with torch.no_grad():
            resnet_features = self.resnet_module(img)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        final_features = self.batchnorm(self.linear_layer(resnet_features))
        return final_features
    
    
class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers, max_sequence_len=20):
        super(LSTM, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, vocab_size)
        self.max_sequence_len = max_sequence_len
        
    def forward(self, in_features, captions, caption_length):
        embeddings = self.embedding_layer(captions)
        embeddings = torch.cat((in_features.unsqueeze(1), embeddings), dim=1)
        lstm_input = pack_padded_sequence(embeddings, caption_length, batch_first=True)
        hidden, _ = self.lstm_layer(lstm_input)
        outputs = self.linear_layer(hidden[0])
        return outputs
    
    def sample(self, in_features, lstm_states=None):
        sampled_idx = []
        lstm_inputs = in_features.unsqueeze(1)
        for i in range(self.max_sequence_len):
            hidden, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)
            outputs = self.linear_layer(hidden.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_idx.append(predicted)
            lstm_inputs = self.embedding_layer(predicted)
            lstm_inputs = lstm_inputs.unsqueeze(1)
        sampled_idx = torch.stack(sampled_idx, 1)
        return sampled_idx