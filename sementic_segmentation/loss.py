from torch import nn


class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        loss = nn.CrossEntropyLoss()(outputs[0], targets)
        loss_aux = nn.CrossEntropyLoss()(outputs[1], targets)
        return loss + self.aux_weight*loss_aux