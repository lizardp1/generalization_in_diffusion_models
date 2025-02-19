import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """Simple MLP for score matching on 2D GMM data"""
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=4):
        super(SimpleNet, self).__init__()
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer - score is same dimension as input
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure input is properly shaped
        if len(x.shape) == 3:  # If input comes as [B, 1, 2]
            x = x.squeeze(1)
        return self.net(x)

def init_weights(m):
    """Initialize weights using kaiming normal"""
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)