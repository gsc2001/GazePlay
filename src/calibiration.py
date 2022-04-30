import cv2
import numpy as np
from gaze_models.gaze_capture.lib.runner import GazeCaptureRunner
from face_eye_detectors.vila_jones import VialaJonesDetector
from process import check_face_eyes


def get_calibration_matrix(
    model_runner: GazeCaptureRunner, face_eye_detector: VialaJonesDetector
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
    #
    cv2.setWindowProperty("Hi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # screen_points = np.array([
    #     [700, 300],
    #     [700, 780],
    #     [1220, 300],
    #     [1220, 780],
    # ], dtype=int)
    # screen_points = np.array(
    #     [
    #         [200, 200],
    #         [1720, 200],
    #         [1720, 880],
    #         [200, 880],
    #     ],
    #     dtype=int,
    # )
    screen_points = np.array(
        [
            [100, 100],
            [960, 100],
            [1820, 100],
            [530, 540],
            [1390, 540],
            [100, 980],
            [960, 980],
            [1820, 980],
        ],
        dtype=int,
    )
    cnt = 0
    points_detected = []
    points_temp = []
    current_point = 0
    mean_pts_n = 3
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
                cv2.imwrite(f"image_{current_point}_{cnt}.png", frame)
                output = model_runner.run(img, faces_eyes)
                points_temp.append(output)
                output[0, 1] *= -1
                cnt += 1
                if cnt >= mean_pts_n:
                    cnt = 0
                    current_point += 1
                    points_detected.append(
                        np.array(points_temp).reshape(mean_pts_n, 2).mean(axis=0)
                    )
                    points_temp = []

                if current_point >= len(screen_points):
                    break
            else:
                print("OOPS")

        if key == ord("q"):
            break

    points_detected = np.array(points_detected, dtype=np.float32)
    screen_points = screen_points.astype(np.float32)
    print(points_detected)
    print(screen_points)
    # cv2.destroyAllWindows()
    p_mat = cv2.findHomography(points_detected, screen_points, method=cv2.RANSAC)[0]
    return p_mat
