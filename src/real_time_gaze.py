import cv2
from face_eye_detectors.vila_jones import VialaJonesDetector
from process import extract_face_eyes


def main():
    face_eye_detector = VialaJonesDetector()

    cap = cv2.VideoCapture(2)
    cv2.namedWindow('Output', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Face', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('eyeL', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('eyeR', cv2.WINDOW_GUI_NORMAL)
    while True:
        ret, frame = cap.read()
        img = frame.copy()
        if not ret:
            print('ERROR! no video --- break')
            break

        faces_eyes = face_eye_detector.detect(frame)

        for (face, eyes) in faces_eyes:
            cv2.rectangle(frame, face[:2], face[2:], (255, 0, 0), 2)
            for eye in eyes:
                cv2.rectangle(frame, eye[:2], eye[2:], (0, 255, 0), 2)

        # check only 1 face and 2 eyes
        if len(faces_eyes) != 1:
            continue

        # face, eye extraction
        eyes_bboxs = faces_eyes[0][1]
        if len(eyes_bboxs) != 2:
            continue
        face, eyeL, eyeR = extract_face_eyes(img, faces_eyes[0])

        cv2.imshow('Output', frame)
        cv2.imshow('Face', face)
        cv2.imshow('eyeL', eyeL)
        cv2.imshow('eyeR', eyeR)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
