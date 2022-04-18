import os

import torch

from .model import ITrackerModel

file_dir = os.path.dirname(os.path.abspath(__file__))


class GazeCaptureRunner:
    def __init__(self):
        self.model = ITrackerModel()
        weights_path = os.path.join(file_dir, 'checkpoint.pth.tar')
        print('Loading weights')
        checkpoint = torch.load(weights_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Weights Loaded!')

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def run(self, face: torch.Tensor, eyeL: torch.Tensor, eyeR: torch.Tensor, grid: torch.Tensor):
        if torch.cuda.is_available():
            face = face.cuda()
            eyeL = eyeL.cuda()
            eyeR = eyeR.cuda()
            grid = grid.cuda()
        return self.model(face, eyeL, eyeR, grid)
