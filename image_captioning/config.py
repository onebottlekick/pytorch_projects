import torch
from torchvision import transforms


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001

IMG_SIZE = (256, 256)
TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

EMBEDDING_SIZE = 256
HIDDEN_SIZE = 512

TRAIN_LOG_PERIOD = 10

MODEL_SAVE_DIR = 'models'
MODEL_SAVE_PERIOD = 1000