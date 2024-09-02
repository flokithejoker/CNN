import torch
import torch.nn as nn
import torch.nn.functional as F


# Do not use transfer learning, you should create a model from scratch. You can of course research published models, but do not copy 1:1
# I did not copy, nonetheless for clarity here is the paper that influenced my architecture decissions.
# An Optimized Architecture of Image Classification Using Convolutional Neural Network
# I.J. Image, Graphics and Signal Processing, 2019, 10, 30-39

# input images are grayscaled with shape (1, H, W) where H and W are 100 so (1, 100, 100)

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        # Input (1, 100, 100)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # After convolutiion (32, 100, 100)
        self.pool = nn.MaxPool2d(2, 2)
        # After pooling (32, 50, 50)

        # Depth increases
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # After 2. convolution (64, 50, 50)
        self.pool = nn.MaxPool2d(2, 2)
        # After pooling (64, 25, 25)

        # Depth increases
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # After 3. convolution (128, 25, 25)
        self.pool = nn.MaxPool2d(2, 2)
        # After pooling (128, 12, 12)
        
        # Now we pass our abstract features from the convolutions to a fully connected layer
        self.fc1 = nn.Linear(128 * 12 * 12, 512) 
        # from 128 * 12 * 12 = 18432 input features we map to 512

        # Finally our output layer will map to 20 different neurons where each represents one of our labels
        self.output = nn.Linear(512, 20)

    def forward(self, x):
        # Convolution 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Convolution 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Convolution 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten - for input to fully connected NN
        x = torch.flatten(x, 1)

        # Apply fully connected layer
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.output(x)

        return x
