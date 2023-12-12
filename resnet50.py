
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# from torchsummary import summary


# # Load training data
# training_set = np.load('training_set.npy')            #,shape (192, 128, 128, 22)
# training_tool = np.load('training_tool.npy')          #concatenate with the embedding of the convlayer ,shape(192, 4)
# training_action = np.load('training_action.npy')      #concatenate with the embedding of the convlayer,shape(192, 4)
# training_labels_1 = np.load('training_labels_1.npy')   # input is training set and action output is 4 tool,shape(192,)LabelEncoder()0-3
# training_labels_2 = np.load('training_labels_2.npy')   # input is training set and tool output is 4 action,shape(192,)LabelEncoder()0-3
# training_labels_3 = np.load('training_labels_3.npy')   # input is training set, output is action and tools 16 combination,shape(192,) LabelEncoder()0-15

# # Load validation data
# validation_set = np.load('validation_set.npy')            #,shape(64, 128, 128, 22)
# validation_tool = np.load('validation_tool.npy')          #concatenate with the embedding of the convlayer,shape(64,4)
# validation_action = np.load('validation_action.npy')      #concatenate with the embedding of the convlayer,shape(64,4)
# validation_labels_1 = np.load('validation_labels_1.npy')  # input is validation set and action output is 4 tool,shape(64,)LabelEncoder()0-3
# validation_labels_2 = np.load('validation_labels_2.npy')  # input is validation set and tool output is 4 action,shape(64,)LabelEncoder()0-3
# validation_labels_3 = np.load('validation_labels_3.npy')  # input is validation set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15

# # Load test data
# test_set = np.load('test_set.npy')            # ,shape(64, 128, 128, 22)
# test_tool = np.load('test_tool.npy')          #concatenate with the embedding of the convlayer,shape(64,4)
# test_action = np.load('test_action.npy')      #concatenate with the embedding of the convlayer,shape(64,4)
# test_labels_1 = np.load('test_labels_1.npy')  # input is test set and action output is 4 tool,shape(64,)LabelEncoder()0-3
# test_labels_2 = np.load('test_labels_2.npy')  # input is test set and tool output is 4 action,shape(64,)LabelEncoder()0-3
# test_labels_3 = np.load('test_labels_3.npy')  # input is test set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15


training_set = np.load('training_set.npy')            #,shape (192, 128, 128, 22)
training_labels_3 = np.load('training_labels_3.npy')   # input is training set, output is action and tools 16 combination,shape(192,) LabelEncoder()0-15

validation_set = np.load('validation_set.npy')            #,shape(64, 128, 128, 22)
validation_labels_3 = np.load('validation_labels_3.npy')  # input is validation set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15
test_set = np.load('test_set.npy')            # ,shape(64, 128, 128, 22)
test_labels_3 = np.load('test_labels_3.npy')  # input is test set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15

# Load the pre-trained model
model = models.resnet50(pretrained=False)

# Modify the first convolution layer
model.conv1 = torch.nn.Conv2d(22, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the last layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)



# Convert the data to PyTorch tensors and create datasets
training_set = torch.tensor(training_set).permute(0, 3, 1, 2).float()
training_labels_3 = torch.tensor(training_labels_3).long()
validation_set = torch.tensor(validation_set).permute(0, 3, 1, 2).float()
validation_labels_3 = torch.tensor(validation_labels_3).long()
test_set = torch.tensor(test_set).permute(0, 3, 1, 2).float()
test_labels_3 = torch.tensor(test_labels_3).long()

train_dataset = TensorDataset(training_set, training_labels_3)
valid_dataset = TensorDataset(validation_set, validation_labels_3)
test_dataset = TensorDataset(test_set, test_labels_3)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Number of epochs
epochs = 150
best_epoch = 0
lowest_val_loss = float('inf')
highest_val_acc = 0
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)
    epoch_val_acc = 100 * correct / total
    epoch_val_loss = running_val_loss / len(valid_dataset)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training loss: {running_loss/len(train_dataset)}")
    print(f"Validation loss: {running_val_loss/len(valid_dataset)}")
    print(f"Validation accuracy: {epoch_val_acc}%")
    if epoch_val_acc > highest_val_acc:
        print(f"New best epoch: {epoch + 1}")
        best_epoch = epoch + 1
        highest_val_acc = epoch_val_acc

        # Save the model
        torch.save(model.state_dict(), 'best_model_toolaction.pth')

# Test phase
model.load_state_dict(torch.load('best_model_toolaction.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")