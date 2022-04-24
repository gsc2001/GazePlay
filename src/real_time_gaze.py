import cv2
from face_eye_detectors.vila_jones import VialaJonesDetector
from process import extract_face_eyes, get_gaze_image
from gaze_models.gaze_capture.data_prep import preprocess
from gaze_models.gaze_capture.runner import GazeCaptureRunner
from PIL import Image

import numpy as np


def main():
    face_eye_detector = VialaJonesDetector()
    model_runner = GazeCaptureRunner()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Output', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Gaze', cv2.WINDOW_GUI_NORMAL)
    while True:
        ret, frame = cap.read()
        # frame = frame[:, ::-1, :]
        # frame = cv2.flip(frame, 1)
        img = frame.copy()
        if not ret:
            print('ERROR! no video --- break')
            break

        faces_eyes = face_eye_detector.detect(frame)

        for (face, eyes) in faces_eyes:
            cv2.rectangle(frame, face[:2], face[2:], (255, 0, 0), 2)
            for eye in eyes:
                cv2.rectangle(frame, eye[:2], eye[2:], (0, 255, 0), 2)

        cv2.imshow('Output', frame)

        # check only 1 face and 2 eyes
        to_run = True
        if len(faces_eyes) != 1:
            to_run = False

        # face, eye extraction
        if to_run:
            eyes_bboxs = faces_eyes[0][1]
            if len(eyes_bboxs) != 2:
                to_run = False

        if to_run:
            face, eyeL, eyeR = extract_face_eyes(img, faces_eyes[0])
            face = Image.fromarray(face[..., ::-1].astype(np.uint8), 'RGB')
            eyeL = Image.fromarray(eyeL[..., ::-1].astype(np.uint8), 'RGB')
            eyeR = Image.fromarray(eyeR[..., ::-1].astype(np.uint8), 'RGB')
            face, eyeL, eyeR, grid = preprocess(face, eyeL, eyeR, faces_eyes, frame.shape)
            output = model_runner.run(face, eyeL, eyeR, grid)
            print(output)
            gaze_image = get_gaze_image(output.detach().cpu().numpy())
            cv2.imshow('Gaze', gaze_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
