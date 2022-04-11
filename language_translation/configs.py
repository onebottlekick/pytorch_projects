import torch


N_EPOCHS = 10
CLIP = 1
BATCH_SIZE = 128
MODEL_SAVE_PATH = 'models/seq2seq.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5