#! python3

import time
import logging
import torch
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter

from architectures import *
from dataLoader import DataLoader

# logging.disable(level=logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s -- %(asctime)s -- %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self, architecture, learning_rate, batch_size, load_path=None, 
                 trainPath=f'./data/balanced_train.csv', 
                 testPath=f'./data/balanced_test.csv'):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.writer = SummaryWriter(f'runs/{str(architecture)}_{time.time()}', 
                                    comment=f'{str(architecture)}--{batch_size=}--{learning_rate=}')

        self.model = architecture
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True) 

        self.trainloader = DataLoader(trainPath, batch_size=self.batch_size)
        self.testloader = DataLoader(testPath, batch_size=10000)

        self.loss = None
        self.epoch = 0

        self.loadModel(load_path)
    
    def train(self, epochs):
        total_batches = int(len(self.trainloader)/self.batch_size)
        
        pbar = tqdm(total=epochs-self.epoch, desc='Epoch: ')
        while self.epoch <= epochs:

            running_loss = 0.0
            
            for i, batch in tqdm(enumerate(self.trainloader), total=total_batches, desc='Batch: '):

                features, labels = self.process_data(batch)
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                log_probs = self.model(features)
                self.loss = self.loss_function(log_probs, labels)
                self.loss.backward()
                self.optimizer.step()
                
                running_loss += self.loss.item()
                
                if i % 100 == 99:
                    self.model.eval()
                    log_probs = self.model(features)
                    print(log_probs)
                    self.model.train()
                    accuracy = self.accuracy(log_probs, labels)
                    
                    loss = running_loss / 100
                    logging.debug(f'{self.epoch=}, {i=}, {loss=}, accuracy={accuracy.item()}')
                    self.writer.add_scalars('Training Loss', {'Loss': loss}, self.epoch * total_batches + i)
                    self.writer.add_scalars('Training Accuracy', {'Accuracy': accuracy.item()}, self.epoch * total_batches + i)
                    
                    test_acc, test_loss = self.predict()
                    self.writer.add_scalars('Testing Loss', {'Loss': test_loss}, self.epoch * total_batches + i)
                    self.writer.add_scalars('Testing Accuracy', {'Accuracy': test_acc}, self.epoch * total_batches + i)
                    self.writer.flush()

                    running_loss = 0.0
                    
                if i != 0 and i % 10000 == 0:
                    self.saveModel(self.epoch, i, name=str(self.model))

            self.trainloader.reset()
            self.epoch += 1
            pbar.update(1)

        self.saveModel(self.epoch, i, name=str(self.model))

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
    
    def predict(self):
        self.testloader.reset()
            
        losses = []
        accuracies = []
        with torch.no_grad():
            for batch in self.testloader:
                features, labels = self.process_data(batch)
                log_probs = self.model(features)

                loss = self.loss_function(log_probs, labels)
                losses.append(loss)

                accuracy = self.accuracy(log_probs, labels)
                accuracies.append(accuracy)
                
        return torch.stack(accuracies).mean().item(), torch.stack(losses).mean().item()
                
    @staticmethod
    @torch.no_grad()
    def accuracy(outputs, labels):
        outputs = torch.round(outputs)
        # logging.debug(torch.unique(outputs, return_counts=True))
        inaccuracies = torch.sum(torch.abs(outputs - labels))
        return 1 - inaccuracies / labels.size(0)
    
    def process_data(self, data):
        """Sends the data to the GPU if available."""
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels
   
    
if __name__ == "__main__":
    # for ml_model in [LogisticRegression, NN0, NN1, NN2]:
    # for ml_model in [NN0, NN1, NN2, NN3]:
    # for ml_model in [NN0d, NN1d, NNLR]:
    # for ml_model in [NN2d, NN3d]:
    for ml_model in [LogisticRegression, NNLR, NN4d]:
        logging.debug(f'\nTraining model: {str(ml_model)}\n')
        architecture = ml_model(119, 1).to(device)
        model = Model(architecture, learning_rate=0.001, batch_size=20000)
        model.train(epochs=50)
        test_accuracy = model.predict()
        print(test_accuracy)
        
    