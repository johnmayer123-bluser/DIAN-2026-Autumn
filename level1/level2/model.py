import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes = 10, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=(5,5),padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(6,16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84,num_classes),
        )

    def forward(self, X):
        return self.classifier(self.features(X))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)