import os
import cv2
import numpy as np

current_file_path = os.path.dirname(os.path.abspath(__file__))
face_model_default = os.path.join(current_file_path, 'haarcascade_frontalface_alt.xml')
eye_model_default = os.path.join(current_file_path, 'haarcascade_eye_tree_eyeglasses.xml')
print(face_model_default)

face_detector_args_default = {}

eye_detector_args_default = {}


class VialaJonesDetector:
    def __init__(self, face_model_path=face_model_default, eye_model_path=eye_model_default):
        self.face_detector = cv2.CascadeClassifier(face_model_path)
        self.eye_detector = cv2.CascadeClassifier(eye_model_path)

    def detect(self, img):
        """
        Detect faces and eyes
        :param img: bgr image
        :return: [[face1, [eye1,eye2..]],[face2,[eye1,eye2..]]..]
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_img)

        data = []

        for face in faces:
            x, y, w, h = face
            faceROI = gray_img[y:y + h, x:x + w]
            eyes = self.eye_detector.detectMultiScale(faceROI)
            eyes_data = []

            for (ex, ey, ew, eh) in eyes:
                eyes_data.append([ex + x, ey + y, ex + x + ew, ey + y + eh])
            data.append([np.array([x, y, x + w, y + h]), np.array(eyes_data)])

        return data
