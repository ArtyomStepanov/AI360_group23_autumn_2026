import tensorflow as tf
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TrainConfig:
    optimizer: torch.optim.optimizer
    batch_size: int
    epoch_count: int
    is_warm_start: bool
    logger
     



class AbstractModel(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    def fit(config: TrainConfig):
        pass

    @abstractmethod
    def forward(self, x):
        pass

class CNN(AbstractModel):
    def __init__(self, nb_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU(inplace=True)

        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.norm4 = nn.BatchNorm2d(num_features=64)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.norm5 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.ReLU(inplace=True)

        self.max_pool6 = nn.MaxPool2d(2)

        self.flatten7 = nn.Flatten()
        self.linear7 = nn.Linear(in_features=1600, out_features=512)
        self.norm7 = nn.BatchNorm1d(num_features=512)
        self.relu7 = nn.ReLU(inplace=True)
        self.linear8 = nn.Linear(in_features=512, out_features=nb_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        self.relu2(x)

        x = self.max_pool3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        self.relu4(x)

        x = self.conv5(x)
        x = self.norm5(x)
        self.relu5(x)

        x = self.max_pool6(x)

        x = self.flatten7(x)
        x = self.linear7(x)
        x = self.norm7(x)
        self.relu(7)
        x = self.linear8(x)
        return x

class ShallowNN(AbstractModel):
    def __init__(self, nb_classes=10):
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


def DNN(AbstractModel):
    def __init__(self, nb_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)


        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout(0.4)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(0.4)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.drop8 = nn.Dropout(0.4)

        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(inplace=True)
        self.drop9 = nn.Dropout(0.4)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(inplace=True)
        self.drop11 = nn.Dropout(0.4)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=True)
        self.drop12 = nn.Dropout(0.4)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.drop14 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.norm14 = nn.BatchNorm1d(512)
        self.relu14 = nn.ReLU(inplace=True)
        self.drop15 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, nb_classes)

    def forward(self, x):
        x = self.drop1(self.relu1(self.norm1(self.conv1(x))))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.pool1(x)

        x = self.drop3(self.relu3(self.norm3(self.conv3(x))))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.pool2(x)

        x = self.drop5(self.relu5(self.norm5(self.conv5(x))))
        x = self.drop6(self.relu6(self.norm6(self.conv6(x))))
        x = self.relu7(self.norm7(self.conv7(x)))
        x = self.pool3(x)

        x = self.drop8(self.relu8(self.norm8(self.conv8(x))))
        x = self.drop9(self.relu9(self.norm9(self.conv9(x))))
        x = self.relu10(self.norm10(self.conv10(x)))
        x = self.pool4(x)

        x = self.drop11(self.relu11(self.norm11(self.conv11(x))))
        x = self.drop12(self.relu12(self.norm12(self.conv12(x))))
        x = self.relu13(self.norm13(self.conv13(x)))
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.drop14(x)
        x = self.fc1(x)
        x = self.relu14(self.norm14(x))
        x = self.drop15(x)
        x = self.fc2(x)
        return x