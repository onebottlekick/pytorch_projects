import torch.nn as nn


def train(model, train_loader, optimizer, epoch):
    total_loss = 0
    
    for i, data in enumerate(train_loader):
        image, label = data
        optimizer.zero_grad()
        outputs = model(image)
        loss = nn.CrossEntropyLoss()(outputs, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i+1)%1000 == 0:
            print(f'[Epoch {epoch + 1}, batch {i+1}] loss={total_loss/1000}')
            total_loss = 0