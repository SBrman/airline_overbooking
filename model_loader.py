#! python3
import torch
from dataLoader import DataLoader
from architectures import NN4d, NNLR, LogisticRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Constants
INPUT_SIZE = 119
OUTPUT_SIZE = 1


class Model:
    def __init__(self, net, loadPath):
        self.model = net
        self.loadModel(loadPath)

    def loadModel(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def predict(self, features):
        with torch.no_grad():
            return self.model(features)
    
    def __call__(self, features):
        return self.predict(features)
        

if __name__ == "__main__":
    testloader = DataLoader(f'./data/balanced_test.csv', batch_size=2)
    network = NN4d(INPUT_SIZE, OUTPUT_SIZE).to(device)
    nn = Model(network, './final_models/Neural_network3d_51_179_t_1670230822.552445.pt')
    X = next(testloader)[0][0]

    X = X.to_dense().to(device)
    print(nn(X))
