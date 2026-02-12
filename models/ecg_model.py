import torch
import torch.nn as nn

class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        # Convolutional blocks
        self.features = nn.Sequential(
            # Conv1: 1 input channel -> 16 output channels
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Conv2: 16 input channels -> 32 output channels
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Fully connected classifier
        # Input features: 32 channels * 125 length (from 500 input) = 4000
        self.classifier = nn.Sequential(
            nn.Linear(4000, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, x):
        # Expect input shape: (batch_size, 1, 500)
        # If input is (batch_size, 500), add channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
