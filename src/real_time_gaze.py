import cv2
from face_eye_detectors.vila_jones import VialaJonesDetector
from process import extract_face_eyes, get_gaze_image, check_face_eyes
from gaze_models.gaze_capture.lib.data_prep import preprocess
from gaze_models.gaze_capture.lib.runner import GazeCaptureRunner
from calibiration import get_calibration_matrix
from PIL import Image

import numpy as np


def main():
    face_eye_detector = VialaJonesDetector()
    model_runner = GazeCaptureRunner()
    p_mat = get_calibration_matrix(model_runner, face_eye_detector)
    print(p_mat)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Gaze", cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        # frame = frame[:, ::-1, :]
        frame = cv2.flip(frame, 1)
        img = frame.copy()
        if not ret:
            print("ERROR! no video --- break")
            break

        faces_eyes = face_eye_detector.detect(frame)

        for (face, eyes) in faces_eyes:
            cv2.rectangle(frame, face[:2], face[2:], (255, 0, 0), 2)
            for eye in eyes:
                cv2.rectangle(frame, eye[:2], eye[2:], (0, 255, 0), 2)

        to_run = check_face_eyes(faces_eyes)

        if to_run:
            output = model_runner.run(img, faces_eyes)
            output = np.vstack((output.reshape(2, 1), np.array([[1]])))
            output[2] *= -1
            # print(output)
            screen_output = p_mat @ output
            screen_output /= screen_output[2]
            screen_output = screen_output.astype(int).reshape(3)
            # print(screen_output)
            if (
                screen_output[1] >= 1080
                or screen_output[1] < 0
                or screen_output[0] >= 1920
                or screen_output[0] < 0
            ):
                # print("OUT OF BOUNDS!")
                continue
            gaze_image = get_gaze_image(screen_output)
            cv2.imshow("Gaze", gaze_image)

        cv2.imshow("Output", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
