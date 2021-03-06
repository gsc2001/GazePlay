from copyreg import pickle
from tkinter import E
import cv2
# from face_eye_detectors.vila_jones import VialaJonesDetector
from face_eye_detectors.dlib_detector.detector import Shape68Detector
from process import extract_face_eyes, get_gaze_image, check_face_eyes, SCREEN_RES
from gaze_models.gaze_capture.lib.data_prep import preprocess
from gaze_models.gaze_capture.lib.runner import GazeCaptureRunner
from calibiration import get_calibration_matrix
from PIL import Image
import sys
import pickle

import numpy as np
import zmq

def main():
    context = zmq.Context()
    #  Socket to talk to server
    print("Creating a server…")
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:5555')

    face_eye_detector = Shape68Detector(eye_size=(90, 50))
    model_runner = GazeCaptureRunner(feature_only=True)
    if len(sys.argv) >= 2:
        sc_x = pickle.load(open('scx.pkl', 'rb'))
        sc_y = pickle.load(open('scy.pkl', 'rb'))
        regressor_x = pickle.load(open('rgx.pkl', 'rb'))
        regressor_y = pickle.load(open('rgy.pkl', 'rb'))

    else:
        sc_x, sc_y, regressor_x, regressor_y = get_calibration_matrix(model_runner, face_eye_detector)
    # print(p_mat)

    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow("Gaze", cv2.WINDOW_GUI_NORMAL)
    # cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    input("Press Enter after starting unity:")

    old_output = None
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

        cv2.imshow("Output", frame)

        if to_run:
            output = model_runner.run(img, faces_eyes)
            if output is None:
                output = old_output
            if output is None:
                print('oopsie')
                continue
            output = output.reshape(1, -1)
            # output = sc_Input.transform(output)
            # output = np.vstack((output.reshape(2, 1), np.array([[1]])))
            # print(output)
            x_coord = regressor_x.predict(output)
            x_coord = sc_x.inverse_transform(x_coord.reshape(1, -1))
            y_coord = regressor_y.predict(output)

            y_coord = sc_y.inverse_transform(y_coord.reshape(1, -1))
            # screen_output = p_mat @ output
            # screen_output /= screen_output[2]
            screen_output = np.array([x_coord[0][0], y_coord[0][0]], dtype=int)

            # screen_output = screen_output[:2].astype(int).reshape(2)
            # print(screen_output)
            # if (
            #         screen_output[1] >= 1080
            #         or screen_output[0] >= 1920
            #         or screen_output[1] < 0
            #         or screen_output[0] < 0
            # ):
            #     # print("OUT OF BOUNDS!")
            #     continue
            screen_output = np.clip(screen_output, [20, 20], SCREEN_RES - 20)
            screen_output = screen_output.astype('float')
            
            gaze_image, screen_output, center = get_gaze_image(screen_output.astype(int), old_output, False)

            center[0] /= SCREEN_RES[0]            
            center[1] /= SCREEN_RES[1]  
                 
            socket.send(bytes(f"message {center[0]} {center[1]}", "utf-8"))

            # cv2.putText(
            #     gaze_image,
            #     f"Predited: {screen_output[0]/SCREEN_RES[0]}, {screen_output[1]/SCREEN_RES[1]}",
            #     SCREEN_RES // 2,
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,  
            #     (255, 0, 0),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.imshow("Gaze", gaze_image)
            # old_output = screen_output.copy()

        key = cv2.waitKey(5)
        if key == ord("q"):
            break

    socket.close()
    context.term()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
