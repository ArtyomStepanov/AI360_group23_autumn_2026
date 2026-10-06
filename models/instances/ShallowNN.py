from torch import nn

from models.abs_model import AbstractModel


class ShallowNN(AbstractModel):
    def __init__(self, nb_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4096, out_features=384)
        self.norm3 = nn.BatchNorm1d(num_features=384)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=384, out_features=192)
        self.norm4 = nn.BatchNorm1d(num_features=192)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=192, out_features=nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x
