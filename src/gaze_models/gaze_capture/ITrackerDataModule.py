import os

import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import transforms
from .data_prep import load_metadata, face_eye_preprocess

current_dir = os.path.dirname(os.path.abspath(__file__))


def robust_metadata_load(meta_file):
    if not os.path.isfile(meta_file):
        raise RuntimeError(f"There is no file as {meta_file}. Wrong data format!")
    metadata = load_metadata(meta_file)
    if metadata is None:
        raise RuntimeError(f'Could not read metafile: {meta_file}')
    return metadata


def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
    except OSError:
        raise RuntimeError(f'Could not read image: {path}')

    return img


class ITrackerData(data.Dataset):
    def __init__(self, data_path, split='train', im_size=(224, 224), grid_size=(25, 25)):
        self.data_path = data_path
        self.im_size = im_size
        self.grid_size = grid_size
        meta_file = os.path.join(data_path, 'metadata.mat')

        self.metadata = robust_metadata_load(os.path.join(self.data_path, 'metadata.mat'))

        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:, 0]
        print(f'loaded split {split} with {len(self.indices)} records')

    def make_grid(self, params):
        grid_len = self.grid_size[0] * self.grid_size[1]
        grid = np.zeros([grid_len, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(grid_len)])
        indsX = np.array([i % self.gridSize[0] for i in range(grid_len)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (
            self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (
            self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (
            self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

        imFace = load_image(imFacePath)
        imEyeL = load_image(imEyeLPath)
        imEyeR = load_image(imEyeRPath)

        img_face, img_eyeL, img_eyeR = face_eye_preprocess(imFace, imEyeL, imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

        face_grid = self.make_grid(self.metadata['labelFaceGrid'][index, :])

        row = torch.LongTensor([int(index)])
        face_grid = torch.FloatTensor(face_grid)
        gaze = torch.FloatTensor(gaze)
        return row, img_face, img_eyeL, img_eyeR, face_grid, gaze
