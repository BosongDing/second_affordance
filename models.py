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



class CustomResnet18_centercam_2net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet18_centercam_2net, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet18(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(num_ftrs*2, num_classes)  # New fully connected layer for concatenated outputs
        self.fc2 = torch.nn.Linear(num_ftrs*2, num_classes)  # New fully connected layer for concatenated outputs

    def forward(self, x): #input shape (batch_size, 6, 128, 128)
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        x1 = self.resnet(inputs5)
        x2 = self.resnet1(inputs6)
        x_c = torch.cat((x1, x2), dim=1)
        x1 = self.fc1(x_c)  # Use the new fully connected layer
        x2 = self.fc2(x_c)  # Use the new fully connected layer
        return x1, x2
    
class CustomResnet18_3cam_3net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet18_3cam_3net, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet18(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = models.resnet18(pretrained=False)
        self.resnet2.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.resnet2.fc = torch.nn.Identity() 
        self.fc1 = torch.nn.Linear(num_ftrs*6, num_classes)
        self.fc2 = torch.nn.Linear(num_ftrs*6, num_classes)  # New fully connected layer for second output

    def forward(self, x): 
        inputs1 = x[:,12:15,:,:]
        inputs2 = x[:,15:18,:,:]
        inputs3 = x[:,6:9,:,:]
        inputs4 = x[:,9:12,:,:]
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        
        x1 = self.resnet(inputs1)
        x2 = self.resnet(inputs2)
        x_r = torch.cat((x1, x2), dim=1)
        x3 = self.resnet1(inputs3)
        x4 = self.resnet1(inputs4)
        x_l = torch.cat((x3,x4), dim=1)
        x5 = self.resnet2(inputs5)
        x6 = self.resnet2(inputs6)
        x_c = torch.cat((x5, x6), dim=1)
        x = torch.cat((x_l,x_r, x_c), dim=1)
        x1 = self.fc1(x)  # Use the new fully connected layer for first output
        x2 = self.fc2(x)  # Use the new fully connected layer for second output
        return x1, x2
    
class CustomResnet18_3cam_6net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet18_3cam_6net, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet18(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = models.resnet18(pretrained=False)
        self.resnet2.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet3 = models.resnet18(pretrained=False)
        self.resnet3.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet4 = models.resnet18(pretrained=False)
        self.resnet4.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet5 = models.resnet18(pretrained=False)
        self.resnet5.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.resnet2.fc = torch.nn.Identity()
        self.resnet3.fc = torch.nn.Identity()
        self.resnet4.fc = torch.nn.Identity()
        self.resnet5.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(num_ftrs*6, num_classes)
        self.fc2 = torch.nn.Linear(num_ftrs*6, num_classes)  # New fully connected layer for second output

    def forward(self, x): 
        inputs1 = x[:,12:15,:,:]
        inputs2 = x[:,15:18,:,:]
        inputs3 = x[:,6:9,:,:]
        inputs4 = x[:,9:12,:,:]
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        
        x1 = self.resnet(inputs1)
        x2 = self.resnet1(inputs2)
        x_r = torch.cat((x1, x2), dim=1)
        x3 = self.resnet2(inputs3)
        x4 = self.resnet3(inputs4)
        x_l = torch.cat((x3,x4), dim=1)
        x5 = self.resnet4(inputs5)
        x6 = self.resnet5(inputs6)
        x_c = torch.cat((x5, x6), dim=1)
        x = torch.cat((x_l,x_r, x_c), dim=1)
        x1 = self.fc1(x)  # Use the new fully connected layer for first output
        x2 = self.fc2(x)  # Use the new fully connected layer for second output
        return x1, x2



    
    
class CustomResnet50_centercam_2net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet50_centercam_2net, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet50(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(num_ftrs*2, num_classes)  # New fully connected layer for concatenated outputs
        self.fc2 = torch.nn.Linear(num_ftrs*2, num_classes)  # New fully connected layer for concatenated outputs

    def forward(self, x): #input shape (batch_size, 6, 128, 128)
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        x1 = self.resnet(inputs5)
        x2 = self.resnet1(inputs6)
        x_c = torch.cat((x1, x2), dim=1)
        x1 = self.fc1(x_c)  # Use the new fully connected layer
        x2 = self.fc2(x_c)  # Use the new fully connected layer
        return x1, x2
    
class CustomResnet50_3cam_3net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet50_3cam_3net, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet50(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = models.resnet50(pretrained=False)
        self.resnet2.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.resnet2.fc = torch.nn.Identity() 
        self.fc1 = torch.nn.Linear(num_ftrs*6, num_classes)
        self.fc2 = torch.nn.Linear(num_ftrs*6, num_classes)  # New fully connected layer for second output

    def forward(self, x): 
        inputs1 = x[:,12:15,:,:]
        inputs2 = x[:,15:18,:,:]
        inputs3 = x[:,6:9,:,:]
        inputs4 = x[:,9:12,:,:]
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        
        x1 = self.resnet(inputs1)
        x2 = self.resnet(inputs2)
        x_r = torch.cat((x1, x2), dim=1)
        x3 = self.resnet1(inputs3)
        x4 = self.resnet1(inputs4)
        x_l = torch.cat((x3,x4), dim=1)
        x5 = self.resnet2(inputs5)
        x6 = self.resnet2(inputs6)
        x_c = torch.cat((x5, x6), dim=1)
        x = torch.cat((x_l,x_r, x_c), dim=1)
        x1 = self.fc1(x)  # Use the new fully connected layer for first output
        x2 = self.fc2(x)  # Use the new fully connected layer for second output
        return x1, x2
    
class CustomResnet50_3cam_6net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet50_3cam_6net, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet50(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = models.resnet50(pretrained=False)
        self.resnet2.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet3 = models.resnet50(pretrained=False)
        self.resnet3.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet4 = models.resnet50(pretrained=False)
        self.resnet4.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet5 = models.resnet50(pretrained=False)
        self.resnet5.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.resnet2.fc = torch.nn.Identity()
        self.resnet3.fc = torch.nn.Identity()
        self.resnet4.fc = torch.nn.Identity()
        self.resnet5.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(num_ftrs*6, num_classes)
        self.fc2 = torch.nn.Linear(num_ftrs*6, num_classes)  # New fully connected layer for second output

    def forward(self, x): 
        inputs1 = x[:,12:15,:,:]
        inputs2 = x[:,15:18,:,:]
        inputs3 = x[:,6:9,:,:]
        inputs4 = x[:,9:12,:,:]
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        
        x1 = self.resnet(inputs1)
        x2 = self.resnet1(inputs2)
        x_r = torch.cat((x1, x2), dim=1)
        x3 = self.resnet2(inputs3)
        x4 = self.resnet3(inputs4)
        x_l = torch.cat((x3,x4), dim=1)
        x5 = self.resnet4(inputs5)
        x6 = self.resnet5(inputs6)
        x_c = torch.cat((x5, x6), dim=1)
        x = torch.cat((x_l,x_r, x_c), dim=1)
        x1 = self.fc1(x)  # Use the new fully connected layer for first output
        x2 = self.fc2(x)  # Use the new fully connected layer for second output
        return x1, x2



    
    


class CustomResnet101_centercam_2net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet101_centercam_2net, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet101(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(num_ftrs*2, num_classes)  # New fully connected layer for concatenated outputs
        self.fc2 = torch.nn.Linear(num_ftrs*2, num_classes)  # New fully connected layer for concatenated outputs

    def forward(self, x): #input shape (batch_size, 6, 128, 128)
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        x1 = self.resnet(inputs5)
        x2 = self.resnet1(inputs6)
        x_c = torch.cat((x1, x2), dim=1)
        x1 = self.fc1(x_c)  # Use the new fully connected layer
        x2 = self.fc2(x_c)  # Use the new fully connected layer
        return x1, x2
    
class CustomResnet101_3cam_3net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet101_3cam_3net, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet101(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = models.resnet101(pretrained=False)
        self.resnet2.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.resnet2.fc = torch.nn.Identity() 
        self.fc1 = torch.nn.Linear(num_ftrs*6, num_classes)
        self.fc2 = torch.nn.Linear(num_ftrs*6, num_classes)  # New fully connected layer for second output

    def forward(self, x): 
        inputs1 = x[:,12:15,:,:]
        inputs2 = x[:,15:18,:,:]
        inputs3 = x[:,6:9,:,:]
        inputs4 = x[:,9:12,:,:]
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        
        x1 = self.resnet(inputs1)
        x2 = self.resnet(inputs2)
        x_r = torch.cat((x1, x2), dim=1)
        x3 = self.resnet1(inputs3)
        x4 = self.resnet1(inputs4)
        x_l = torch.cat((x3,x4), dim=1)
        x5 = self.resnet2(inputs5)
        x6 = self.resnet2(inputs6)
        x_c = torch.cat((x5, x6), dim=1)
        x = torch.cat((x_l,x_r, x_c), dim=1)
        x1 = self.fc1(x)  # Use the new fully connected layer for first output
        x2 = self.fc2(x)  # Use the new fully connected layer for second output
        return x1, x2
    
class CustomResnet101_3cam_6net(torch.nn.Module):
    def __init__(self, num_classes=4): 
        super(CustomResnet101_3cam_6net, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet1 = models.resnet101(pretrained=False)
        self.resnet1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = models.resnet101(pretrained=False)
        self.resnet2.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet3 = models.resnet101(pretrained=False)
        self.resnet3.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet4 = models.resnet101(pretrained=False)
        self.resnet4.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet5 = models.resnet101(pretrained=False)
        self.resnet5.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity() 
        self.resnet1.fc = torch.nn.Identity()
        self.resnet2.fc = torch.nn.Identity()
        self.resnet3.fc = torch.nn.Identity()
        self.resnet4.fc = torch.nn.Identity()
        self.resnet5.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(num_ftrs*6, num_classes)
        self.fc2 = torch.nn.Linear(num_ftrs*6, num_classes)  # New fully connected layer for second output

    def forward(self, x): 
        inputs1 = x[:,12:15,:,:]
        inputs2 = x[:,15:18,:,:]
        inputs3 = x[:,6:9,:,:]
        inputs4 = x[:,9:12,:,:]
        inputs5 = x[:,3:6,:,:]
        inputs6 = x[:,0:3,:,:]
        
        x1 = self.resnet(inputs1)
        x2 = self.resnet1(inputs2)
        x_r = torch.cat((x1, x2), dim=1)
        x3 = self.resnet2(inputs3)
        x4 = self.resnet3(inputs4)
        x_l = torch.cat((x3,x4), dim=1)
        x5 = self.resnet4(inputs5)
        x6 = self.resnet5(inputs6)
        x_c = torch.cat((x5, x6), dim=1)
        x = torch.cat((x_l,x_r, x_c), dim=1)
        x1 = self.fc1(x)  # Use the new fully connected layer for first output
        x2 = self.fc2(x)  # Use the new fully connected layer for second output
        return x1, x2
