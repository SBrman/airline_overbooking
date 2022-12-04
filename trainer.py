#! python3

import time
import torch
import logging
from tqdm import tqdm, trange
from architectures import LogisticRegression, NN1, NN2, NN3
from dataLoader import DataLoader

# logging.disable(level=logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s -- %(asctime)s -- %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self, architecture, learning_rate, batch_size, load_path=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = architecture
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) 
        self.loss = None
        self.loadModel(load_path)
    
    def train(self, epochs, trainPath):
        trainloader = DataLoader(trainPath, batch_size=self.batch_size)
        for epoch in trange(epochs, desc="Epochs: "):
            for i, batch in tqdm(enumerate(trainloader), desc='Batch: '):
                features, labels = self.process_data(batch)
                self.model.zero_grad()
                log_probs = self.model(features)
                self.loss = self.loss_function(log_probs, labels)
                self.loss.backward()
                self.optimizer.step()
                
                if i % 1000 == 0:
                    accuracy = self.accuracy(log_probs, labels)
                    logging.debug(f'{epoch=}, {i=}, loss={self.loss.item()}, accuracy={accuracy.item()}')
                    
                if i != 0 and i % 10000 == 0:
                    self.saveModel(epoch, i, name=str(self.model))

    def saveModel(self, epoch, batch_num, name='model'):
        infoDict = {
            'epoch': epoch, 
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }
        torch.save(infoDict, fr'./model/{name}_{epoch}_{batch_num}_t_{time.time()}.pt')
        
    def loadModel(self, path):
        if path is None:
            return
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.model.eval()
    
    def predict(self, testPath):
        testloader = DataLoader(testPath, batch_size=self.batch_size)

        losses = []
        accuracies = []
        with torch.no_grad():
            for batch in testloader:
                features, labels = self.process_data(batch)
                log_probs = self.model(features)

                loss = self.loss_function(log_probs, labels)
                losses.append(loss)

                accuracy = self.accuracy(log_probs, labels)
                accuracies.append(accuracy)
                
        return torch.stack(accuracies).mean().item()
                
    @staticmethod
    @torch.no_grad()
    def accuracy(outputs, labels):
        outputs.round()
        inaccuracies = torch.sum(torch.abs(outputs - labels))
        return 1 - inaccuracies / labels.size(0)
    
    def process_data(self, data):
        """Sends the data to the GPU if available."""
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels
    
if __name__ == "__main__":
    for ml_model in [LogisticRegression, NN1, NN2, NN3]:
        logging.debug(f'\nTraining model: {str(ml_model)}\n')
        architecture = ml_model(119, 1).to(device)
        model = Model(architecture, learning_rate=0.0001, batch_size=2048)
        model.train(epochs=10, trainPath=f'./data/balanced_train.csv')
        model.predict(testPath=f'./data/balanced_test.csv')
        
    