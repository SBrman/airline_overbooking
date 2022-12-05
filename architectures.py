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


class NNLR(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNLR, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, dtype=torch.float64)
        self.fc2 = nn.Linear(input_size, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.fc2(output)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "NNLR"


class NN0(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN0, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network0"


class NN0d(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN0d, self).__init__()
        self.fc1 = nn.Linear(input_size, 254, dtype=torch.float64)
        self.fc2 = nn.Linear(254, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.fc2(output)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network0d"
    

class NN1(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network1"


class NN1d(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN1d, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.fc3(output)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network1d"
    
    
class NN2(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 1024, dtype=torch.float64)
        self.fc3 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc4 = nn.Linear(512, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        output = self.dropout(F.relu(self.fc4(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network2"


class NN2d(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN2d, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 1024, dtype=torch.float64)
        self.fc3 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc4 = nn.Linear(512, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        output = self.fc4(output)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network2d"
    

class NN3(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN3, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc3 = nn.Linear(256, 64, dtype=torch.float64)
        self.fc4 = nn.Linear(64, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        output = self.dropout(F.relu(self.fc4(output)))
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network3"


class NN3d(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN3d, self).__init__()
        self.fc1 = nn.Linear(input_size, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc3 = nn.Linear(256, 64, dtype=torch.float64)
        self.fc4 = nn.Linear(64, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        output = self.fc4(output)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network3d"



class NN4d(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN4d, self).__init__()
        self.fc1 = nn.Linear(input_size, 256, dtype=torch.float64)
        self.fc2 = nn.Linear(256, 128, dtype=torch.float64)
        self.fc3 = nn.Linear(128, 64, dtype=torch.float64)
        self.fc4 = nn.Linear(64, output_size, dtype=torch.float64)
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, feature):
        output = self.dropout(F.relu(self.fc1(feature)))
        output = self.dropout(F.relu(self.fc2(output)))
        output = self.dropout(F.relu(self.fc3(output)))
        output = self.fc4(output)
        return torch.sigmoid(output)
    
    def __str__(self):
        return "Neural_network3d"