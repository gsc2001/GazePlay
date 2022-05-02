import os

import cv2
import dlib
import numpy as np

current_file_path = os.path.dirname(os.path.abspath(__file__))
face_model_default = os.path.join(current_file_path, 'shape_predictor_68_face_landmarks.dat')


class Shape68Detector:
    def __init__(self, eye_size=(50, 50)):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor(face_model_default)
        self.left_idx = [36, 37, 38, 39, 40, 41]
        self.right_idx = [42, 43, 44, 45, 46, 47]
        self.eye_size = np.array(eye_size)

    def get_eye_coords(self, shape):
        eyeL = np.array([(shape.part(idx).x, shape.part(idx).y) for idx in self.left_idx]).reshape(6, 2)
        eyeR = np.array([(shape.part(idx).x, shape.part(idx).y) for idx in self.right_idx]).reshape(6, 2)
        return eyeL.mean(axis=0), eyeR.mean(axis=0)

    def detect(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_img, 1)

        data = []

        for face in faces:
            shape = self.landmark_detector(gray_img, face)
            eyeL_center, eyeR_center = self.get_eye_coords(shape)
            eyes_data = [[*(eyeL_center - self.eye_size // 2), *(eyeL_center + self.eye_size // 2)],
                         [*(eyeR_center - self.eye_size // 2), *(eyeR_center + self.eye_size // 2)]]

            data.append([np.array([face.tl_corner().x, face.tl_corner().y, face.br_corner().x, face.br_corner().y]),
                         np.array(eyes_data, dtype=int)])

        return data
