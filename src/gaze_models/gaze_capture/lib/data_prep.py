import os

import numpy as np
import scipy.io as sio

from torchvision.transforms import transforms
import torch


class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.meanImg)


def get_face_grid(grid_shape, face_bbox, img_shape):
    grid = np.zeros(grid_shape, dtype=np.float32)
    face_bbox[0] *= grid_shape[0] / img_shape[0]
    face_bbox[2] *= grid_shape[0] / img_shape[0]
    face_bbox[1] *= grid_shape[1] / img_shape[1]
    face_bbox[3] *= grid_shape[1] / img_shape[1]
    x, y, x2, y2 = np.array(face_bbox, dtype=int)
    grid[y:y2, x:x2] = 1
    return grid


def load_metadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        return None
    return metadata


metadata_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
face_mean = load_metadata(os.path.join(metadata_dir, 'mean_face_224.mat'))['image_mean']
left_mean = load_metadata(os.path.join(metadata_dir, 'mean_left_224.mat'))['image_mean']
right_mean = load_metadata(os.path.join(metadata_dir, 'mean_right_224.mat'))['image_mean']

face_transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    SubtractMean(meanImg=face_mean)
])

eyeL_transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    SubtractMean(meanImg=left_mean)
])

eyeR_transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    SubtractMean(meanImg=right_mean)
])


def face_eye_preprocess(face, eyeL, eyeR):
    return face_transformation(face), eyeL_transformation(eyeL), eyeR_transformation(eyeR)


def preprocess(face, eyeL, eyeR, face_eyes_bbox, img_shape):
    """
    Preprocess face, eyeL and eyeR to run iTracker on it also return the face grid
    """

    transformed_face, transformed_eyeL, transformed_eyeR = face_eye_preprocess(face, eyeL, eyeR)

    grid = get_face_grid((25, 25), face_eyes_bbox[0][0], img_shape)

    face = torch.reshape(transformed_face, (1, 3, 224, 224)).float()
    eyeL = torch.reshape(transformed_eyeL, (1, 3, 224, 224)).float()
    eyeR = torch.reshape(transformed_eyeR, (1, 3, 224, 224)).float()
    grid = torch.FloatTensor(grid).reshape((1, 625))

    return face, eyeL, eyeR, grid
