import cv2
import numpy as np


def crop(img, bbox):
    x, y, x2, y2 = bbox
    return img[y:y2, x:x2]


def extract_face_eyes(img, face_eyes_bbox):
    """
    Return face and eyeL and eyeR
    :param img:
    :param face_eyes: [face, [eye1, eye2]]
    :return: [face, eyeL, eyeR]
    """

    face_bbox = face_eyes_bbox[0]
    eyeL_bbox = face_eyes_bbox[1][0]
    eyeR_bbox = face_eyes_bbox[1][1]

    if eyeL_bbox[0] > eyeR_bbox[0]:
        # swap them
        eyeL_bbox, eyeR_bbox = eyeR_bbox, eyeL_bbox

    return [crop(img, face_bbox), crop(img, eyeL_bbox), crop(img, eyeR_bbox)]


def get_gaze_image(model_output):
    img = np.zeros((1080, 1920, 3))
    cv2.circle(img, (model_output[0], model_output[1]), 10, (255, 255, 255), -1)
    return img


def get_gaze_image_old(model_output, img_size=400):
    img = np.zeros((img_size, img_size, 3))
    model_output[1] *= -1
    pixel_point = (model_output * 10 + img_size // 2).astype(int).squeeze()
    print(pixel_point)
    cv2.circle(img, (pixel_point[0], pixel_point[1]), 10, (255, 255, 255), -1)
    return img


def check_face_eyes(faces_eyes):
    to_run = True
    if len(faces_eyes) != 1:
        to_run = False

    # face, eye extraction
    if to_run:
        eyes_bboxs = faces_eyes[0][1]
        if len(eyes_bboxs) != 2:
            to_run = False

    return to_run
