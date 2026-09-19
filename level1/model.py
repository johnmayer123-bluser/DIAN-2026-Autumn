import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_dim = 784, hidden_dims=(256,128),num_classes=10,dropout=0.1):
        super().__init__()
        layers = [nn.Flatten()]
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev_dim,hidden),nn.ReLU(),nn.Dropout(dropout)]
            prev_dim = hidden

        layers.append(nn.Linear(prev_dim,num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self,X):
        return self.net(X)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)