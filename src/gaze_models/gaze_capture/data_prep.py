import numpy as np


def get_face_grid(grid_shape, face_bbox, img_shape):
    grid = np.zeros(grid_shape, dtype=np.float32)
    face_bbox[0] *= grid_shape[0] / img_shape[0]
    face_bbox[2] *= grid_shape[0] / img_shape[0]
    face_bbox[1] *= grid_shape[1] / img_shape[1]
    face_bbox[3] *= grid_shape[1] / img_shape[1]
    x, y, x2, y2 = np.int(face_bbox)
    grid[y:y2, x:x2] = 1
    return grid


def preprocess(face, eyeL, eyeR, face_eyes_bbox, img_shape):
    """
    Preprocess face, eyeL and eyeR to run iTracker on it also return the face grid
    """
    # TODO: implement this!
    pass
