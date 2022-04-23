import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from constants import *


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 224 x 224 x 3
            nn.Conv2d(
                3,
                CONV1_OUT,
                kernel_size=CONV1_KERNEL,
                stride=CONV1_STRIDE,
                padding=CONV1_PADDING,
            ),
            # 54 x 54 x 96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=MAX_POOL1_KERNEL, stride=MAX_POOL1_STRIDE),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            # 26 x 26 x 96
            nn.Conv2d(
                CONV1_OUT,
                CONV2_OUT,
                kernel_size=CONV2_KERNEL,
                stride=CONV2_STRIDE,
                padding=CONV2_PADDING,
                groups=2,
            ),
            # 26 x 26 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=MAX_POOL2_KERNEL, stride=MAX_POOL2_STRIDE),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            # 12 x 12 x 256
            nn.Conv2d(
                CONV2_OUT,
                CONV3_OUT,
                kernel_size=CONV3_KERNEL,
                stride=CONV3_STRIDE,
                padding=CONV3_PADDING,
            ),
            # 12 x 12 x 384
            nn.ReLU(inplace=True),
            nn.Conv2d(
                CONV3_OUT,
                CONV4_OUT,
                kernel_size=CONV4_KERNEL,
                stride=CONV4_STRIDE,
                padding=CONV4_PADDING,
            ),
            # 12 x 12 x 64
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.reshape(x.size(0), -1)
        return x


class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CNNModel()
        self.fc = nn.Sequential(
            # 12 * 12 * 64
            nn.Linear(FINAL_CNN_DIM, FC_F1),
            nn.ReLU(inplace=True),
            # 128
            nn.Linear(FC_F1, FC_F2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.fc = nn.Sequential(
            # 25 * 25
            nn.Linear(GRID_SIZE * GRID_SIZE, FC_FG1),
            nn.ReLU(inplace=True),
            # 256
            nn.Linear(FC_FG1, FC_FG2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eyeModel = CNNModel()
        self.faceModel = FaceModel()
        self.gridModel = FaceGridModel()

        self.eyesFC = nn.Sequential(
            nn.Linear(2 * FINAL_CNN_DIM, FC_E1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(FC_E1 + FC_F2 + FC_FG2, FC1),
            nn.ReLU(inplace=True),
            nn.Linear(FC1, FC2),
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)

        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)

        return x
