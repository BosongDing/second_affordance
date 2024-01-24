import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
from models import CustomResnet18_3cam_6net
# from torchsummary import summary
metrics = {
    'train_loss': [],
    'train_accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'test_precision': None,
    'test_recall': None,
    'test_accuracy': None,
    'test_f1': None,
    'confusion_matrix': None
}
argparser = argparse.ArgumentParser()
argparser.add_argument('--random_seed', type=int, default=3407, help='Model to use')
args = argparser.parse_args()

# Set the random seed
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# # Load training data
# training_set = np.load('training_set.npy')            #,shape (192, 128, 128, 22)
# training_tool = np.load('training_tool.npy')          #concatenate with the embedding of the convlayer ,shape(192, 4)
# training_action = np.load('training_action.npy')      #concatenate with the embedding of the convlayer,shape(192, 4)
training_labels_1 = np.load('../training_labels_1.npy')   # input is training set and action output is 4 tool,shape(192,)LabelEncoder()0-3
training_labels_2 = np.load('../training_labels_2.npy')   # input is training set and tool output is 4 action,shape(192,)LabelEncoder()0-3
# training_labels_3 = np.load('training_labels_3.npy')   # input is training set, output is action and tools 16 combination,shape(192,) LabelEncoder()0-15

# # Load validation data
# validation_set = np.load('validation_set.npy')            #,shape(64, 128, 128, 22)
# validation_tool = np.load('validation_tool.npy')          #concatenate with the embedding of the convlayer,shape(64,4)
# validation_action = np.load('validation_action.npy')      #concatenate with the embedding of the convlayer,shape(64,4)
validation_labels_1 = np.load('../validation_labels_1.npy')  # input is validation set and action output is 4 tool,shape(64,)LabelEncoder()0-3
validation_labels_2 = np.load('../validation_labels_2.npy')  # input is validation set and tool output is 4 action,shape(64,)LabelEncoder()0-3
# validation_labels_3 = np.load('validation_labels_3.npy')  # input is validation set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15

# # Load test data
# test_set = np.load('test_set.npy')            # ,shape(64, 128, 128, 22)
# test_tool = np.load('test_tool.npy')          #concatenate with the embedding of the convlayer,shape(64,4)
# test_action = np.load('test_action.npy')      #concatenate with the embedding of the convlayer,shape(64,4)
test_labels_1 = np.load('../test_labels_1.npy')  # input is test set and action output is 4 tool,shape(64,)LabelEncoder()0-3
test_labels_2 = np.load('../test_labels_2.npy')  # input is test set and tool output is 4 action,shape(64,)LabelEncoder()0-3
# test_labels_3 = np.load('test_labels_3.npy')  # input is test set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15

file_name = os.path.splitext(os.path.basename(__file__))[0]
                    
best_model_path = file_name + str(args.random_seed)+'.pth'
training_set = np.load('../training_set.npy')[:,:,:,:]     #,shape (192, 128, 128, 22)
training_labels_3 = np.load('../training_labels_3.npy')   # input is training set, output is action and tools 16 combination,shape(192,) LabelEncoder()0-15

validation_set = np.load('../validation_set.npy')[:,:,:,:]            #,shape(64, 128, 128, 22)
validation_labels_3 = np.load('../validation_labels_3.npy')  # input is validation set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15
test_set = np.load('../test_set.npy')[:,:,:,:]            # ,shape(64, 128, 128, 22)
test_labels_3 = np.load('../test_labels_3.npy')  # input is test set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15

# Load the pre-trained model
model = CustomResnet18_3cam_6net()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)



# Convert the data to PyTorch tensors and create datasets
training_set = torch.tensor(training_set).permute(0, 3, 1, 2).float()
training_labels_2 = torch.tensor(training_labels_2).long()
training_labels_1 = torch.tensor(training_labels_1).long()
training_labels_3 = torch.tensor(training_labels_3).long()
validation_set = torch.tensor(validation_set).permute(0, 3, 1, 2).float()
validation_labels_3 = torch.tensor(validation_labels_3).long()
validation_labels_2 = torch.tensor(validation_labels_2).long()
validation_labels_1 = torch.tensor(validation_labels_1).long()
test_set = torch.tensor(test_set).permute(0, 3, 1, 2).float()
test_labels_2 = torch.tensor(test_labels_2).long()
test_labels_3 = torch.tensor(test_labels_3).long()
test_labels_1 = torch.tensor(test_labels_1).long()
train_dataset = TensorDataset(training_set,training_labels_1,training_labels_2)
valid_dataset = TensorDataset(validation_set,validation_labels_1,validation_labels_2)
test_dataset = TensorDataset(test_set,test_labels_1,test_labels_2)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Number of epochs
epochs = 150
best_epoch = 0
lowest_val_loss = float('inf')
highest_val_acc = 0
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, labels1, labels2 in train_loader:
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs)
        loss1 = criterion(outputs1, labels1)
        loss2 = criterion(outputs2, labels2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels1, labels2 in valid_loader:
            inputs = inputs.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)

            outputs1, outputs2 = model(inputs)
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            total += labels1.size(0)
            correct += ((predicted1 == labels1) & (predicted2 == labels2)).sum().item()
            loss1 = criterion(outputs1, labels1)
            loss2 = criterion(outputs2, labels2)
            loss = loss1 + loss2

            running_val_loss += loss.item() * inputs.size(0)
    epoch_val_acc = 100 * correct / total
    epoch_val_loss = running_val_loss / len(valid_dataset)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training loss: {running_loss/len(train_dataset)}")
    print(f"Validation loss: {running_val_loss/len(valid_dataset)}")
    print(f"Validation accuracy: {epoch_val_acc}%")
    metrics['train_loss'].append(running_loss/len(train_dataset))
    metrics['train_accuracy'].append(100 * correct / total)
    metrics['val_loss'].append(epoch_val_loss)
    metrics['val_accuracy'].append(epoch_val_acc)
    if epoch_val_acc > highest_val_acc:
        print(f"New best epoch: {epoch + 1}")
        best_epoch = epoch + 1
        highest_val_acc = epoch_val_acc

        # Save the model
        torch.save(model.state_dict(), best_model_path)

# Test phase
model.load_state_dict(torch.load(best_model_path))
model.eval()
y_true1 = []
y_pred1 = []
y_true2 = []
y_pred2 = []
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels1, labels2 in test_loader:
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        outputs1, outputs2 = model(inputs)
        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        total += labels1.size(0)
        correct += ((predicted1 == labels1) & (predicted2 == labels2)).sum().item()
        y_true1.extend(labels1.cpu().numpy())
        y_pred1.extend(predicted1.cpu().numpy())
        y_true2.extend(labels2.cpu().numpy())
        y_pred2.extend(predicted2.cpu().numpy())

print(f"Test Accuracy: {100 * correct / total}%")
metrics['test_accuracy'] = 100 * correct / total
metrics['test_precision_label1'] = precision_score(y_true1, y_pred1, average='macro')
metrics['test_recall_label1'] = recall_score(y_true1, y_pred1, average='macro')
metrics['test_f1_label1'] = f1_score(y_true1, y_pred1, average='macro')
metrics['confusion_matrix_label1'] = confusion_matrix(y_true1, y_pred1)

metrics['test_precision_label2'] = precision_score(y_true2, y_pred2, average='macro')
metrics['test_recall_label2'] = recall_score(y_true2, y_pred2, average='macro')
metrics['test_f1_label2'] = f1_score(y_true2, y_pred2, average='macro')
metrics['confusion_matrix_label2'] = confusion_matrix(y_true2, y_pred2)

print(f"Test Accuracy: {100 * correct / total}%")
file_name = os.path.splitext(os.path.basename(__file__))[0]
path_name  = file_name + str(args.random_seed)+'.pkl'
with open(path_name, 'wb') as f:
    pickle.dump(metrics, f)