import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        self.conv_l1 = nn.Conv2d(1, 40, 5)
        self.conv_l2 = nn.Conv2d(40, 40, 5)

        # For Step 1
        self.fc_s1 = nn.Linear(28*28, 100)
        self.fc1_s1 = nn.Linear(100,10) #10 Classifiers in O/P

        #For Step 2
        self.fc_s2 = nn.Linear(40*4*4, 100) #Dimension of Image after Pooling

        #For Step 4
        self.fc_s4 = nn.Linear(100, 100)

        #For Step 5
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(40* 4* 4, 1000)
        self.fc1 = nn.Linear(1000,1000)
        self.fc2 = nn.Linear(1000,10)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        X = X.view(-1, 28 * 28 * 1)
        X = F.sigmoid(self.fc_s1(X))
        return self.fc1_s1(X)

    # Use two convolutional layers.
    def model_2(self, X):
        X = F.sigmoid(self.conv_l1(X))
        X = F.max_pool2d(X, 2)
        X = F.sigmoid(self.conv_l2(X))
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = F.sigmoid(self.fc_s2(X))
        return self.fc1_s1(X)


    # Replace sigmoid with ReLU(Step 3).
    def model_3(self, X):
        
        X = F.relu(self.conv_l1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv_l2(X))
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = F.relu(self.fc_s2(X))
        return self.fc1_s1(X)


    # Add one extra fully connected layer.
    def model_4(self, X):
        X = F.relu(self.conv_l1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv_l2(X))
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = F.relu(self.fc_s2(X))
        X = self.fc_s4(X)
        return self.fc1_s1(X)

    # Use Dropout now.
    def model_5(self, X):
        X = F.relu(self.conv_l1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv_l2(X))
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = self.dropout(X)
        X = F.relu(self.fc(X))
        X = F.relu(self.fc1(X))
        return self.fc2(X)