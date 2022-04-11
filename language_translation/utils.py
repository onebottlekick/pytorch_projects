import torch.nn as nn


def tokenize(text, nlp, reverse=False):
    if reverse:
        return [token.text for token in nlp.tokenizer(text)][::-1]
    return [token.text for token in nlp.tokenizer(text)]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = elapsed_time - int(elapsed_mins*60)
    return elapsed_mins, elapsed_secs


def init_weights(module):
    for name, param in module.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)