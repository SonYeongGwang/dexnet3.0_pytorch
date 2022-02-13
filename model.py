import torch

from torch import nn

## GQ-CNN architecture is identical to Dex-Net 2.0 except that the pose input stream /
## to include the angle between the approach direction and the table normal.

class SuctionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        ## shape of the images at the last conv layer is (13x13)
        self.fc1 = nn.Linear(13*13*64, 1024)
        self.act5 = nn.ReLU()
        self.fc2 = nn.Linear(1040, 1024)
        self.act6 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 2)

        self.pose_fc1 = nn.Linear(2, 16)
        self.pose_act1 = nn.ReLU()

    def forward(self, input_depth, input_pose):
        out = self.act1(self.conv1(input_depth))
        out = self.pool1(self.act2(self.conv2(out)))
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))
        out = out.view(-1, 13 * 13 * 64)

        out_pose = self.pose_fc1(input_pose)
        out_pose = self.pose_act1(out_pose)
        
        out = self.act5(self.fc1(out))
        out = torch.cat((out, out_pose), 1)
        out = self.act6(self.fc2(out))
        out = self.fc3(out)
        return out