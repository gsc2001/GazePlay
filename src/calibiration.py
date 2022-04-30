import cv2
import numpy as np
from gaze_models.gaze_capture.lib.runner import GazeCaptureRunner
from face_eye_detectors.vila_jones import VialaJonesDetector
from process import check_face_eyes


def get_calibration_matrix(model_runner: GazeCaptureRunner, face_eye_detector: VialaJonesDetector):
    window = cv2.namedWindow('Hi', cv2.WINDOW_NORMAL)
    #
    cv2.setWindowProperty('Hi', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # screen_points = np.array([
    #     [700, 300],
    #     [700, 780],
    #     [1220, 300],
    #     [1220, 780],
    # ], dtype=int)
    screen_points = np.array([
        [200, 200],
        [1720, 200],
        [1720, 880],
        [200, 880],
    ], dtype=int)
    cnt = 0
    points_detected = []
    points_temp = []
    current_point = 0
    mean_pts_n = 5
    while True:

        ret, frame = cap.read()
        if not ret:
            print('ERROR! no video --- break')
            break
        # frame = cv2.imread(f'image_{current_point}_{cnt}.png')
        output_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(output_img, f'Cnt: {cnt}', (960, 540), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(output_img, screen_points[current_point], 10, (255, 255, 255), -1)

        cv2.imshow('Hi', output_img)

        key = cv2.waitKey(1)
        if key == ord(' '):
            # run model
            img = frame.copy()

            faces_eyes = face_eye_detector.detect(frame)
            to_run = check_face_eyes(faces_eyes)

            if to_run:
                cv2.imwrite(f'image_{current_point}_{cnt}.png', frame)
                output = model_runner.run(img, faces_eyes)
                points_temp.append(output)
                output[0, 1] *= -1
                cnt += 1
                if cnt >= mean_pts_n:
                    cnt = 0
                    current_point += 1
                    points_detected.append(np.array(points_temp).reshape(mean_pts_n, 2).mean(axis=0))
                    points_temp = []

                if current_point >= len(screen_points):
                    break
            else:
                print('OOPS')

        if key == ord('q'):
            break

    points_detected = np.array(points_detected, dtype=np.float32)
    screen_points = screen_points.astype(np.float32)
    print(points_detected)
    print(screen_points)
    # cv2.destroyAllWindows()
    p_mat = cv2.getPerspectiveTransform(points_detected, screen_points)
    return p_mat
