import torch.nn
import numpy as np


class CNN2(torch.nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.keep_prob = 0.5

        n_channels_1 = 10
        n_channels_2 = 20

        self.layer1 = torch.nn.Sequential( \
            torch.nn.Conv2d(1, n_channels_1, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential( \
            torch.nn.Conv2d(n_channels_1, n_channels_2, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc3 = torch.nn.Linear(5 * 5 * n_channels_2, 150, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.layer3 = torch.nn.Sequential( \
            self.fc3,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        self.fc4 = torch.nn.Linear(150, 80, bias=True)
        self.layer4 = torch.nn.Sequential( \
            self.fc4,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        self.fc5 = torch.nn.Linear(80, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x = np.transpose(x, (0, 3, 1, 2))
        x = x.mean(dim=1)
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        out = self.layer3(out)
        out = self.layer4(out)

        out = self.fc5(out)
        return out
