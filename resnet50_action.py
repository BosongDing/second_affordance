
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
# from torchsummary import summary


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(22, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last layer
        self.fc = nn.Linear(num_ftrs + 4, 4)   # New last layer

    def forward(self, x, tool):
        x = self.resnet(x)
        x = torch.cat((x, tool), dim=1)  # Concatenate the tool vector to the output of the conv layers
        x = self.fc(x)
        return x
# # Load training data
# training_set = torch.load('training_set.pt')            #,shape (192, 128, 128, 22)
#training_tool = torch.load('training_tool.pt')          #concatenate with the embedding of the convlayer ,shape(192, 4)
training_action = torch.load('training_action.pt')      #concatenate with the embedding of the convlayer,shape(192, 4)
training_labels_1 = torch.load('training_labels_1.pt')   # input is training set and action output is 4 tool,shape(192,)LabelEncoder()0-3
#training_labels_2 = torch.load('training_labels_2.pt')   # input is training set and tool output is 4 action,shape(192,)LabelEncoder()0-3
# training_labels_3 = torch.load('training_labels_3.pt')   # input is training set, output is action and tools 16 combination,shape(192,) LabelEncoder()0-15

# # Load validation data
# validation_set = torch.load('validation_set.pt')            #,shape(64, 128, 128, 22)
#validation_tool = torch.load('validation_tool.pt')          #concatenate with the embedding of the convlayer,shape(64,4)
validation_action = torch.load('validation_action.pt')      #concatenate with the embedding of the convlayer,shape(64,4)
validation_labels_1 = torch.load('validation_labels_1.pt')  # input is validation set and action output is 4 tool,shape(64,)LabelEncoder()0-3
#validation_labels_2 = torch.load('validation_labels_2.pt')  # input is validation set and tool output is 4 action,shape(64,)LabelEncoder()0-3
# validation_labels_3 = torch.load('validation_labels_3.pt')  # input is validation set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15

# # Load test data
# test_set = torch.load('test_set.pt')            # ,shape(64, 128, 128, 22)
#test_tool = torch.load('test_tool.pt')          #concatenate with the embedding of the convlayer,shape(64,4)
test_action = torch.load('test_action.pt')      #concatenate with the embedding of the convlayer,shape(64,4)
test_labels_1 = torch.load('test_labels_1.pt')  # input is test set and action output is 4 tool,shape(64,)LabelEncoder()0-3
#test_labels_2 = torch.load('test_labels_2.pt')  # input is test set and tool output is 4 action,shape(64,)LabelEncoder()0-3
# test_labels_3 = torch.load('test_labels_3.pt')  # input is test set, output is action and tools 16 combination,shape(64,) LabelEncoder()0-15


training_set = torch.load('training_set.pt')            #, shape (192, 128, 128, 22)
validation_set = torch.load('validation_set.pt')        #, shape(64, 128, 128, 22)
test_set = torch.load('test_set.pt')                    #, shape(64, 128, 128, 22)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Move the model to the device
model = CustomResNet().to(device)

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())



# Convert the data to PyTorch tensors and create datasets
training_set = torch.tensor(training_set).permute(0, 3, 1, 2).float()
training_labels_1 = torch.tensor(training_labels_1).long()
validation_set = torch.tensor(validation_set).permute(0, 3, 1, 2).float()
validation_labels_1 = torch.tensor(validation_labels_1).long()
test_set = torch.tensor(test_set).permute(0, 3, 1, 2).float()
test_labels_1 = torch.tensor(test_labels_1).long()
training_action = torch.tensor(training_action).float()
validation_action = torch.tensor(validation_action).float()
test_action = torch.tensor(test_action).float()



train_dataset = TensorDataset(training_set, training_action, training_labels_1)
valid_dataset = TensorDataset(validation_set, validation_action, validation_labels_1)
test_dataset = TensorDataset(test_set, test_action, test_labels_1)

# Create data loaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Number of epochs
epochs = 100

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, tool, labels in train_loader:
        inputs = inputs.to(device)
        tool = tool.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, tool)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, tools,labels in valid_loader:
            inputs = inputs.to(device)
            tools = tools.to(device)
            labels = labels.to(device)

            outputs = model(inputs,tools)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training loss: {running_loss/len(train_dataset)}")
    print(f"Validation loss: {running_val_loss/len(valid_dataset)}")

# Test phase
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, tools,labels in test_loader:
        inputs = inputs.to(device)
        tools = tools.to(device)
        labels = labels.to(device)

        outputs = model(inputs,tools)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")