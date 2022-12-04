#! python3

import torch
from torch import nn
from torch.functional import F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size, dtype=torch.float64)
        
    def forward(self, feature):
        output = self.linear(feature)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Logistic_regression"


class NN1(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_size, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 256, dtype=torch.float64)
        self.fc3 = nn.Linear(256, 512, dtype=torch.float64)
        self.fc4 = nn.Linear(512, 1024, dtype=torch.float64)
        self.fc5 = nn.Linear(1024, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        output = self.dropout(F.relu(self.fc4(output)))
        output = self.dropout(F.relu(self.fc5(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network1"
    
    
class NN2(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network2"
    
    
class NN3(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN3, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc2 = nn.Linear(256, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network2"