import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from .constants import *
from .model import CNNModel


class FaceLandmarksModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(FACELANDMARKS * 2, FC_FG1_3),
            nn.ReLU(inplace=True),
            # 256
            nn.Linear(FC_FG1_3, FC_FG2_3),
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
        self.landmarksModel = FaceLandmarksModel()

        self.eyesFC = nn.Sequential(
            nn.Linear(2 * FINAL_CNN_DIM, FC_E1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(FC_E1 + FC_FG2_3, FC1),
            nn.ReLU(inplace=True),
            nn.Linear(FC1, FC2),
        )

    def forward(self, eyesLeft, eyesRight, faceLandmarks):
        eyeL = self.eyeModel(eyesLeft)
        eyeR = self.eyeModel(eyesRight)

        eyes = torch.cat((eyeL, eyeR), 1)
        eyes = self.eyesFC(eyes)

        landmarks = self.landmarksModel(faceLandmarks)

        x = torch.cat((eyes, landmarks), 1)
        x = self.fc(x)

        return x
