from abs_models import AbstractModel


class CNN(AbstractModel):
    def __init__(self, nb_classes):
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
        self.relu7(x)
        x = self.linear8(x)
        return x
