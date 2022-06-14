import os
import cv2
import time
import numpy as np
import mediapipe as mp
from numbers import Real
from typing import List, Tuple, Union

# constants
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
LIP_DISTANCE_THRESHOLD = (
    25  # threshold to determine if mouth is open or not (chosen via experimentation)
)

# Landmark indices
UPPER_LIP = [82, 13, 312, 37, 0, 267]
LOWER_LIP = [87, 14, 317, 84, 17, 314]
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# CREDITS FOR THE EYES POSITION ESTIMATION: AiPhile Youtube channel
# I dubbed a huge part of the code from his GitHub repository
# SOURCE: https://github.com/Asadullah-Dal17
def extract_eye_landmarks(face_mesh_landmarks: List[List[Real]]) -> Tuple[list, list]:
    """Extract the landmarks for the right and left eye from the face

    Args:
        face_mesh_landmarks (list): Mediapipe face mesh landmarks

    Returns:
        right_eye_landmarks (list): list of landmark positions for the right eye
        left_eye_landmarks (list): list of landmark positions for the left eye
    """
    right_eye_landmarks = [face_mesh_landmarks[idx] for idx in RIGHT_EYE]
    left_eye_landmarks = [face_mesh_landmarks[idx] for idx in LEFT_EYE]
    return right_eye_landmarks, left_eye_landmarks


def extract_lip_landmarks(face_mesh_landmarks: List[List[Real]]) -> Tuple[list, list]:
    """Extract the upper and lower-lip landmarks from the face

    Args:
        face_mesh_landmarks (list): Mediapipe face mesh landmarks

    Returns:
        upper_lip_points (list): list of landmark positions for the upper lip
        lower_lip_points (list): list of landmarks positions for the lower lip
    """
    upper_lip_points = [face_mesh_landmarks[idx] for idx in UPPER_LIP]
    lower_lip_points = [face_mesh_landmarks[idx] for idx in LOWER_LIP]
    return upper_lip_points, lower_lip_points


def eyes_extractor(
    img: np.ndarray,
    right_eye_coords: List[List[Real]],
    left_eye_coords: List[List[Real]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the right and left eye from the face

    Args:
        img (np.ndarray): Face image
        right_eye_coords (list): right eye landmarks
        left_eye_coords (list): left eye landmarks

    Returns:
        cropped_right (np.ndarray): The extracted right eye
        cropped_left (np.ndarray): The extracted left eye
    """
    # converting color image to  scale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # creating mask from gray scale dim
    mask = np.zeros(gray_img.shape, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # draw eyes image on mask, where white shape is
    eyes = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    # change black color to gray other than eyes
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes
    # For RIGHT Eye
    r_max_x = int((max(right_eye_coords, key=lambda item: item[0]))[0])
    r_min_x = int((min(right_eye_coords, key=lambda item: item[0]))[0])
    r_max_y = int((max(right_eye_coords, key=lambda item: item[1]))[1])
    r_min_y = int((min(right_eye_coords, key=lambda item: item[1]))[1])

    # For LEFT Eye
    l_max_x = int((max(left_eye_coords, key=lambda item: item[0]))[0])
    l_min_x = int((min(left_eye_coords, key=lambda item: item[0]))[0])
    l_max_y = int((max(left_eye_coords, key=lambda item: item[1]))[1])
    l_min_y = int((min(left_eye_coords, key=lambda item: item[1]))[1])

    # croping the eyes from mask
    cropped_right = eyes[r_min_y:r_max_y, r_min_x:r_max_x]
    cropped_left = eyes[l_min_y:l_max_y, l_min_x:l_max_x]

    # returning the cropped eyes
    return cropped_right, cropped_left


# Eyes Postion Estimator
def position_estimator(cropped_eye: np.ndarray) -> str:
    """Estimate the gaze position of the extracted eye

    Args:
        cropped_eye (np.ndarray): cropped eye image array

    Returns:
        eye_position (str): RIGHT | LEFT | CENTER | CLOSED
    """
    # getting height and width of eye
    h, w = cropped_eye.shape

    # remove the noise from images
    gaussain_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv2.medianBlur(gaussain_blur, 3)

    # applying thresholding to convert binary image
    _, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)
    # create fixed part for eye width
    piece = int(w / 3)

    # slicing the eyes into three parts
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece : piece + piece]
    left_piece = threshed_eye[0:h, piece + piece : w]

    # calling pixel counter function
    eye_position = eye_pixel_counter(right_piece, center_piece, left_piece)

    return eye_position


# creating pixel counter function
def eye_pixel_counter(first_piece, second_piece, third_piece):
    """Count the pixels in the cropped eye image to estimate the position"""
    # counting black pixel in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ""
    if max_index == 0:
        pos_eye = "RIGHT"
    elif max_index == 1:
        pos_eye = "CENTER"
    elif max_index == 2:
        pos_eye = "LEFT"
    else:
        pos_eye = "CLOSED"
    return pos_eye


def compute_lip_distance(
    upper_lip_points: List[Real], lower_lip_points: List[Real]
) -> Real:
    """Computes the euclidean distance between the upper lip and lower lip.
    This is used to estimate if the mouth is open or not.

    Args:
        upper_lip_points (List[Real,Real]): upper lip landmark coordinates
        lower_lip_points (List[Real,Real]): lower lip landmark coordinates

    Returns:
        lip_distance (Real): the computed lip distance
    """
    upper_lip_mean = np.mean(upper_lip_points, axis=0)
    lower_lip_mean = np.mean(lower_lip_points, axis=0)
    return np.sqrt(
        (upper_lip_mean[1] - lower_lip_mean[1]) ** 2
        + (upper_lip_mean[0] - lower_lip_mean[0]) ** 2
    )


def compute_face_orientation(
    image: np.ndarray,
    face_points_2d: Union[list, np.ndarray],
    face_points_3d: Union[list, np.ndarray],
    nose_2d: Union[list, np.ndarray, None],
    draw_nose_line: bool = True,
) -> str:
    """Compute the face orientation

    Args:
        img_w (float): Width of the image
        img_h (float): Height of the image
        face_points_2d (np.ndarray): list of 2d face points
        face_points_3d (np.ndarray): list of 3d face points

    Returns:
        orientation (str): RIGHT | LEFT | UP | FORWARD
    """
    img_h, img_w = image.shape[:2]
    # defining the camera matrix
    # with calibration, we can automatically compute the required focal_length and camera_matrix
    focal_length = 1 * img_w
    # Camera matrix is a 3 x 3 matrix taken as input
    camera_matrix = np.array(
        [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
    )

    # define the distortion coefficients
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    # OpenCV solvePnP method
    success, rotational_vector, translational_vector = cv2.solvePnP(
        face_points_3d, face_points_2d, camera_matrix, dist_coeffs, flags=0
    )
    # Get the rotational matrix around the (x,y,z) co-ordinates
    rot_matrix, _ = cv2.Rodrigues(rotational_vector)

    # Get the angles (compute the RQ Decomposition of the rotational matrix)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)

    # Get the x,y,z rotation estimation
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotational_vector, translational_vector, camera_matrix, dist_coeffs)

    # points for drawing the line from the nose
    if draw_nose_line:
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        # draw the line
        cv2.line(image, p1, p2, (255, 0, 0), 3)

    # Estimate the direction of the user's head
    if y > 10:
        direction = "RIGHT"
    elif y < -10:
        direction = "LEFT"
    elif x > 10:
        direction = "UP"
    elif x < -10:
        direction = "DOWN"
    else:
        direction = "FORWARD"

    return direction


def draw_landmarks(image: np.ndarray, face_landmarks) -> None:
    """Draw the computed face landmarks using Mediapipe

    Args:
        image: the input image
        face_landmarks: face landmarks computed by mediapipe
    """
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
    )


if __name__ == "__main__":
    # Head Pose Estimation entails determining the position and orientation of the head
    # The OpenCV solvepnp() method is used to estimate the orientation of a 3D object in a 2D image
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()  # read the video frame
            start = time.time()
            if not success:
                print("Ignoring video frame")
                continue
            # To improve performance, optionally mark the image as not writeable to pass by reference
            image.flags.writeable = False
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True  # enable image writeability, because we will be drawing the face landmarks on the face image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            (img_h, img_w, n_channels) = image.shape

            desired_landmark_idxs = [
                1,
                33,
                61,
                199,
                263,
                291,
            ]  # for the face orientation computation
            # Now, to draw the face mesh annotations on the image
            # if the computed array of face landmarks is non-empty
            if (
                results.multi_face_landmarks
            ):  # iterate through all the faces in the array
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    print(type(face_landmarks))
                    face_points_3d = []
                    face_points_2d = []
                    upper_lip_points = []
                    lower_lip_points = []
                    # There are 468 3D face landmarks according to the documentation
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx in desired_landmark_idxs:
                            if (
                                idx == 1
                            ):  # this is the estimated landmark point for a nose
                                # we need the nose points to draw a line later on
                                nose_2d = (landmark.x * img_w, landmark.y * img_h)
                                nose_3d = (
                                    landmark.x * img_w,
                                    landmark.y * img_h,
                                    landmark.z * 3000,
                                )

                            x, y = int(landmark.x * img_w), int(landmark.y * img_h)

                            # Get the 2D Co-ordinates
                            face_points_2d.append([x, y])
                            face_points_3d.append([x, y, landmark.z])
                    face_mesh_landmarks = [
                        [landmark.x * img_w, landmark.y * img_h]
                        for landmark in face_landmarks.landmark
                    ]

                    face_points_2d = np.array(face_points_2d, dtype=np.float64)
                    face_points_3d = np.array(face_points_3d, dtype=np.float64)
                    upper_lip_points, lower_lip_points = extract_lip_landmarks(
                        face_mesh_landmarks
                    )
                    right_eye_points, left_eye_points = extract_eye_landmarks(
                        face_mesh_landmarks
                    )

                    crop_right_eye, crop_left_eye = eyes_extractor(
                        image, right_eye_points, left_eye_points
                    )
                    eye_position_right = position_estimator(crop_right_eye)
                    eye_position_left = position_estimator(crop_left_eye)

                    print("Eye position right: ", eye_position_right)
                    print("Eye position left: ", eye_position_left)

                    lip_distance = compute_lip_distance(
                        upper_lip_points, lower_lip_points
                    )
                    print("Lip distance: ", lip_distance)

                    face_orientation = compute_face_orientation(
                        image, face_points_2d, face_points_3d, nose_2d
                    )
                    # Add the direction text to the image
                    cv2.putText(
                        image,
                        face_orientation,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        2,
                    )

                    if lip_distance > LIP_DISTANCE_THRESHOLD:
                        cv2.putText(
                            image,
                            f"Mouth {face_idx} is Open !!!",
                            (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )

                    cv2.putText(
                        image,
                        eye_position_right,
                        (20, 400),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3,
                    )
                    draw_landmarks(image, face_landmarks)
            end = time.time()
            fps = int(1 / (end - start))
            print(f"FPS: {fps:>.2f} FPS")
            cv2.putText(
                image,
                f"FPS: {fps}",
                (450, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3,
            )
            cv2.imshow("Mediapipe Face Mesh output", image)
            if cv2.waitKey(1) == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
