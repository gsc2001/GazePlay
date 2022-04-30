import os

from PIL import Image
import numpy as np
import torch

from process import extract_face_eyes
from .data_prep import preprocess

from .model import ITrackerModel

file_dir = os.path.dirname(os.path.abspath(__file__))


class GazeCaptureRunner:
    def __init__(self):
        self.model = ITrackerModel()
        # weights_path = os.path.join(file_dir, '..', '..', '', 'checkpoint_cpu.pth.tar')
        weights_path = 'checkpoint_cpu.pth.tar'

        print('Loading weights')
        checkpoint = torch.load(weights_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        # print(next(self.model.parameters()).is_cuda)
        # self.model = self.model.cpu()
        # checkpoint['state_dict'] = self.model.state_dict()
        # torch.save(checkpoint, 'checkpoint_cpu.pth.tar')
        # print(torch.cuda.is_available())
        print('Weights Loaded!')

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def run(self, img, faces_eyes):
        face, eyeL, eyeR = extract_face_eyes(img, faces_eyes[0])
        face = Image.fromarray(face[..., ::-1].astype(np.uint8), 'RGB')
        eyeL = Image.fromarray(eyeL[..., ::-1].astype(np.uint8), 'RGB')
        eyeR = Image.fromarray(eyeR[..., ::-1].astype(np.uint8), 'RGB')
        face, eyeL, eyeR, grid = preprocess(face, eyeL, eyeR, faces_eyes, img.shape)
        if torch.cuda.is_available():
            face = face.cuda()
            eyeL = eyeL.cuda()
            eyeR = eyeR.cuda()
            grid = grid.cuda()
        return self.model(face, eyeL, eyeR, grid).detach().cpu().numpy()
