import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import cv2
import math

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

EDGES = {
    # (0, 1): 'm',
    # (0, 2): 'c',
    # (1, 3): 'm',
    # (2, 4): 'c',
    # (0, 5): 'm',
    # (0, 6): 'c',
    (5, 7): (255, 0, 0),
    (7, 9): (255, 0, 0),
    # (6, 8): 'c',
    # (8, 10): 'c',
    # (5, 6): 'y',
    (5, 11): (0, 255, 0),
    # (6, 12): 'c',
    # (11, 12): 'y',
    (11, 13): (0, 0, 255),
    (13, 15): (0, 0, 255),
    # (12, 14): 'c',
    # (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     color, 3, cv2.LINE_4)

            # Calculate the angle between the two keypoints
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Draw the angle text on the frame
            cv2.putText(frame, f'{angle:.1f}', (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)


def draw_keypoints(frame, keypoints, confidence_threshold, edges):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, _ in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.circle(frame, (int(x1), int(y1)), 7, (0, 0, 255), -1)
            cv2.circle(frame, (int(x2), int(y2)), 7, (0, 0, 255), -1)

            # Write the joint names on the drawn joints
            joint_name1 = list(KEYPOINT_DICT.keys())[
                list(KEYPOINT_DICT.values()).index(p1)]
            joint_name2 = list(KEYPOINT_DICT.keys())[
                list(KEYPOINT_DICT.values()).index(p2)]
            cv2.putText(frame, joint_name1, (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            cv2.putText(frame, joint_name2, (int(x2), int(y2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)


interpreter = tf.lite.Interpreter(
    model_path='lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()

cap = cv2.VideoCapture(1)
out = None
save_video = False

while cap.isOpened():
    ret, frame = cap.read()

    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Rendering
    draw_connections(frame, keypoints_with_scores, EDGES, 0.2)
    draw_keypoints(frame, keypoints_with_scores, 0.2, EDGES)

    cv2.imshow('MoveNet Thunder', frame)

    if out is None:
        height, width, _ = frame.shape
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
            *'XVID'), 7.5, (width, height))
    out.write(frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
