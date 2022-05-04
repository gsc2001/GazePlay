from copyreg import pickle
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from gaze_models.gaze_capture.lib.runner import GazeCaptureRunner
from process import check_face_eyes, SCREEN_RES
import pickle


def get_calibration_matrix(
        model_runner: GazeCaptureRunner, face_eye_detector
):
    # return np.array(
    #     [
    #         [-1.04859702e02, -3.43884692e01, 1.01383601e03],
    #         [-2.92141565e01, 2.02145111e01, 3.75424645e02],
    #         [-3.93820261e-02, -4.62538636e-02, 1.00000000e00],
    #     ]
    # )
    window = cv2.namedWindow("Hi", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Face", cv2.WINDOW_GUI_NORMAL)

    cv2.setWindowProperty("Hi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    x, y = SCREEN_RES

    screen_points = np.array(
        [
            [50, 50],
            [x // 2, 50],
            [x - 50, 50],
            [x // 4 + 25, y // 4 + 25],
            [3 * x // 4 - 25, y // 4 + 25],
            [50, y // 2],
            [x // 2, y // 2],
            [x - 50, y // 2],
            [x // 4 + 25, 3 * y // 4 - 25],
            [3 * x // 4 - 25, 3 * y // 4 - 25],
            [50, y - 100],
            [x // 2, y - 100],
            [x - 50, y - 100],
        ],
        dtype=int,
    )
    cnt = 0
    # points_detected = []
    # points_temp = []
    input_features = []
    output_values_x = []
    output_values_y = []
    current_point = 0
    mean_pts_n = 5

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        img = frame.copy()
        if not ret:
            print("ERROR! no video --- break")
            break
        # frame = cv2.imread(f'image_{current_point}_{cnt}.png')
        output_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(
            output_img,
            f"Cnt: {cnt}",
            (960, 540),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(output_img, screen_points[current_point], 10, (255, 255, 255), -1)

        cv2.imshow("Hi", output_img)
        faces_eyes = face_eye_detector.detect(frame)

        for (face, eyes) in faces_eyes:
            cv2.rectangle(frame, face[:2], face[2:], (255, 0, 0), 2)
            for eye in eyes:
                cv2.rectangle(frame, eye[:2], eye[2:], (0, 255, 0), 2)
        cv2.imshow("Face", frame)
        key = cv2.waitKey(1)
        if key == ord(" "):
            # run model
            img = img.copy()

            to_run = check_face_eyes(faces_eyes)

            if to_run:
                cv2.imwrite(f"images/image_{current_point}_{cnt}.png", frame)
                output = model_runner.run(img, faces_eyes)
                # points_temp.append(output)
                input_features.append(output)
                output_values_x.append(screen_points[current_point][0])
                output_values_y.append(screen_points[current_point][1])
                cnt += 1
                if cnt >= mean_pts_n:
                    cnt = 0
                    current_point += 1
                    # points_detected.append(
                    #     np.array(points_temp).reshape(mean_pts_n, 2).mean(axis=0)
                    # )
                    # points_temp = []

                if current_point >= len(screen_points):
                    break
            else:
                print("OOPS")

        if key == ord("q"):
            break

    input_features = np.array(input_features, dtype=np.float32)
    output_values_x = np.array(output_values_x, dtype=int).reshape(-1, 1)
    output_values_y = np.array(output_values_y, dtype=int).reshape(-1, 1)
    print(output_values_x, output_values_y)

    # sc_Input = StandardScaler()
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    # Input = sc_Input.fit_transform(input_features)
    x = sc_x.fit_transform(output_values_x)
    y = sc_y.fit_transform(output_values_y)
    regressor_x = SVR(kernel='linear')
    regressor_y = SVR(kernel='linear')
    regressor_x.fit(input_features, x)
    regressor_y.fit(input_features, y)

    pickle.dump(sc_x, open("scx.pkl", 'wb'))
    pickle.dump(sc_y, open("scy.pkl", 'wb'))
    pickle.dump(regressor_x, open("rgx.pkl", 'wb'))
    pickle.dump(regressor_y, open("rgy.pkl", 'wb'))

    # points_detected = np.array(points_detected, dtype=np.float32)
    # screen_points = screen_points.astype(np.float32)
    # print(points_detected)
    # print(screen_points)
    cv2.destroyAllWindows()
    # p_mat, points = cv2.findHomography(points_detected, screen_points, method=cv2.RANSAC)
    # print(points)
    # return sc_Input, sc_x, sc_y, regressor_x, regressor_y
    return sc_x, sc_y, regressor_x, regressor_y
