import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from tqdm import tqdm
import csv

class victim_model():

    def __init__(self, model_type, lr):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == 'mnist':
            self.model = mnist_cnn()#.to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum = 0.5)
            self.loss_fn = nn.CrossEntropyLoss()
        elif model_type == 'cifar10':
            self.model = cifar10_cnn()#.to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum = 0.9)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            assert False

    def train_model(self, epochs, dataset):
      csv_file = 'victim_training_res.csv'
      print(epochs)
      results = []
      for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0
        
        for inputs, labels in tqdm(dataset.train_loader):
        # for inputs, labels in train_loader:
            inputs = inputs#.to(self.device)
            labels = labels#.to(self.device)
    
            outputs = self.model(inputs)
    
            loss = self.loss_fn(outputs, labels) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
    
        else:
            with torch.no_grad(): # No gradient for validation
                for val_inputs, val_labels in tqdm(dataset.test_loader):
                    val_inputs = val_inputs#.to(self.device)
                    val_labels = val_labels#.to(self.device)
    
                    val_outputs = self.model(val_inputs)
    
                    val_loss = self.loss_fn(val_outputs, val_labels)
                    
                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data).item()
                
            epoch_loss = running_loss/len(dataset.train_loader)
            epoch_acc = running_corrects/ len(dataset.train_dataset)
            #epoch_acc = running_corrects/ 60_000
            
            val_epoch_loss = val_running_loss/len(dataset.test_loader)
            val_epoch_acc = val_running_corrects/ len(dataset.test_dataset)
            # val_epoch_acc = val_running_corrects/ 60_000
            
            
            epoch_results = {
                'Epoch': e,
                'Training Loss': round(epoch_loss, 3),
                'Validation Loss': round(val_epoch_loss, 3),
                'Training Accuracy': round(epoch_acc,3),
                'Validation Accuracy': round(val_epoch_acc,3)
                }
            
            print(epoch_results)
            results.append(epoch_results)
    
      csv_columns = ['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy',  'Validation Accuracy']
      try:
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in results:
                    writer.writerow(data)
      except IOError:
            print("I/O error")
      return self.model

class mnist_cnn(nn.Module):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
class cifar10_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
