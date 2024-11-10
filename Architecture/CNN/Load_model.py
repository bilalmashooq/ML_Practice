import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, out_channels1=32, out_channels2=64, out_channels3=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels1,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels2,
            out_channels=out_channels3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(out_channels3 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(out_channels1=32, out_channels2=64, out_channels3=128).to(device)
model.load_state_dict(torch.load('cnn_checkpoint.pth'))
model.eval()
print("Model checkpoint loaded.")
