import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from face_utils import (
    extract_eye_landmarks,
    extract_lip_landmarks,
    eyes_extractor,
    position_estimator,
    compute_lip_distance,
    compute_face_orientation,
    draw_landmarks,
    LIP_DISTANCE_THRESHOLD,
)
from yolov5_utils import detect, draw_bounding_boxes

# mediapipe drawing utilities configuration
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
YOLO_ASSETS_DIR = os.path.join(ASSETS_DIR, "yolo")
# test video
EXAM_PROCTOR_DEMO_VIDEO = os.path.join(ASSETS_DIR, "demo-2.mp4")
# maximum number of faces to detect in each frame
MAX_NUMBER_FACES = 20
# read the classes.txt file
f = open(os.path.join(YOLO_ASSETS_DIR, "classes.txt"), encoding="UTF-8", mode="r+")
# COCO class names
CLASSNAMES = f.read().split("\n")
# default classes to detect
DEFAULT_CLASSES = ["person", "book", "tvmonitor", "cell phone", "bottle", "laptop"]

# Initialize the streamlit app with a title
st.title("Exam Proctor with Streamlit and Mediapipe")
add_sidebar_title = st.sidebar.title("EXAM PROCTOR")
# add_sidebar_subheader = st.sidebar.subheader("")
# Navigation menu
app_mode = st.sidebar.selectbox("Navigation", ["About App", "Run on Video"])

# @st.cache() caches the function in memory
@st.cache()
def image_resize(
    image, width: float = None, height: float = None, inter=cv2.INTER_AREA
):
    """
    Utility function for resizing an image, while preserving its aspect ratio.
    """
    output_dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None and height is not None:
        output_dim = (int(width), int(height))
    elif width is None:
        ratio = height / float(h)
        output_dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        output_dim = (width, int(h * ratio))

    # resize the image
    resized_image = cv2.resize(image, output_dim, interpolation=inter)
    return resized_image


if app_mode == "About App":
    st.markdown(
        """
        This application is an implementation of real-time Face mesh detection using **Streamlit**, **OpenCV**, and **MediaPipe**.
    """
    )
    st.markdown(
        """
        ### Technologies
        - [Mediapipe](https://mediapipe.readthedocs.io/en/latest/)
        - OpenCV
        - [Streamlit](https://docs.streamlit.io/)
        - Python
    """
    )
    st.markdown(
        """
        ### References 
        - Mediapipe documentation
    """
    )

elif app_mode == "Run on Video":
    st.set_option("deprecation.showfileUploaderEncoding", False)
    use_webcam = st.sidebar.button("Use Webcam")
    use_demo_video = st.sidebar.button("Use Demo Video")
    record = st.sidebar.checkbox("Record Video")
    # placeholder for the video stream
    vid = None
    stop = None

    if record:
        st.checkbox("Recording", value=True)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px 
            margin-left: -350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )

    MAX_FACES = st.sidebar.number_input(
        "Maximum Number of Faces", value=MAX_NUMBER_FACES, min_value=1
    )
    st.sidebar.markdown("---")
    DETECTION_CONFIDENCE = st.sidebar.slider(
        "Minimum Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        key="video-slider-dc",
    )
    TRACKING_CONFIDENCE = st.sidebar.slider(
        "Minimum Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        key="video-slider-tc",
    )
    st.sidebar.markdown("---")

    SELECTED_CLASSES = st.sidebar.multiselect(
        "Select objects to detect in stream",
        CLASSNAMES,
        DEFAULT_CLASSES,
    )

    classes_map = dict(zip(range(len(CLASSNAMES)), CLASSNAMES))

    selected_class_idxs = dict(
        filter(lambda x: x[1] in list(SELECTED_CLASSES), classes_map.items())
    ).keys()

    if not vid:
        with st.expander("⚡️ See Usage Instructions"):
            st.markdown(
                """
                ### Video Stream Options 
                - Webcam Feed
                - Video Upload
                - An optional Demo Video is also available

                ⚙️ To get started: 
                - Select video stream option from the sidebar 
                - View the predictions
                - Click the 'Stop Stream' button beneath the video stream to stop the stream
                """
            )

    video_file_buffer = st.sidebar.file_uploader(
        "Upload a Video", type=["mp4", "mov", "avi", "asf", "m4v"]
    )
    tffile = tempfile.NamedTemporaryFile(
        delete=False
    )  # temp placeholder for the video file

    ## Get the input video
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        if use_demo_video:
            vid = cv2.VideoCapture(EXAM_PROCTOR_DEMO_VIDEO)
            tffile.name = EXAM_PROCTOR_DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    if vid:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        # Recording part
        codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        out = cv2.VideoWriter("output1.mp4", codec, fps_input, (width, height))

        st.sidebar.text("Input Video")
        st.sidebar.video(tffile.name)

        drawing_spec = mp_drawing.DrawingSpec(
            color=(0, 128, 0), thickness=1, circle_radius=1
        )

        st.markdown("## Output")

        # Initialize a streamlit frame for displaying the stream
        st_video_frame = st.empty()

        # Button for stopping the video stream
        stop = st.button("Stop Stream")

        # for displaying the frame rate and other info
        info1, info2, info3, info4, info5 = st.columns(5)

        with info1:
            st.write(
                '<h6 style="text-align:center;">Face Count</h6>', unsafe_allow_html=True
            )
            info1_text = st.markdown("0")
        with info2:
            st.write(
                '<h6 style="text-align:center;">Head Orientation</h6>',
                unsafe_allow_html=True,
            )
            info2_text = st.markdown("0")
        with info3:
            st.write(
                '<h6 style="text-align:center;">Mouth</h6>', unsafe_allow_html=True
            )
            info3_text = st.markdown("CLOSED")
        with info4:
            st.write('<h6 style="text-align:center;">Eye</h6>', unsafe_allow_html=True)
            info4_text = st.markdown("0")
        with info5:
            st.write('<h6 style="text-align:center;">FPS</h6>', unsafe_allow_html=True)
            info5_text = st.markdown("0")
        st.markdown("<hr />", unsafe_allow_html=True)

        # Initialize the yolov5 net
        net = cv2.dnn.readNet(os.path.join(YOLO_ASSETS_DIR, "yolov5s.onnx"))

        face_orientation = None
        eye_position_right = None
        eye_position_left = None
        lip_status = None

        with mp_face_mesh.FaceMesh(
            max_num_faces=MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        ) as face_mesh:
            while vid.isOpened():
                start_time = time.time()
                success, frame = vid.read()  # read the video frame
                if not success:
                    print("Ignoring video frame")
                    if use_demo_video:  # if using the demo video, stop the stream
                        st_video_frame = None
                        stop = None
                        vid = None
                        break

                    continue
                face_count = 0

                # To improve performance, optionally mark the image as not writeable to pass by reference
                frame.flags.writeable = False
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
                frame.flags.writeable = True  # enable image writeability, because we will be drawing the face landmarks on the face image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                (img_h, img_w, n_channels) = frame.shape

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
                    for face_idx, face_landmarks in enumerate(
                        results.multi_face_landmarks
                    ):
                        face_count += 1
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

                        try:
                            upper_lip_points, lower_lip_points = extract_lip_landmarks(
                                face_mesh_landmarks
                            )
                            right_eye_points, left_eye_points = extract_eye_landmarks(
                                face_mesh_landmarks
                            )

                            crop_right_eye, crop_left_eye = eyes_extractor(
                                frame, right_eye_points, left_eye_points
                            )
                            eye_position_right = position_estimator(crop_right_eye)
                            eye_position_left = position_estimator(crop_left_eye)

                            # print("Eye position right: ", eye_position_right)
                            # print("Eye position left: ", eye_position_left)

                            lip_distance = compute_lip_distance(
                                upper_lip_points, lower_lip_points
                            )
                            # print("Lip distance: ", lip_distance)

                            lip_status = (
                                "OPEN"
                                if lip_distance > LIP_DISTANCE_THRESHOLD
                                else "CLOSED"
                            )

                            face_orientation = compute_face_orientation(
                                frame, face_points_2d, face_points_3d, nose_2d
                            )

                            # Detect objects
                            result = detect(frame, net)
                            # filter for only selected classes
                            class_idxs = list(
                                filter(lambda x: x in selected_class_idxs, result[0])
                            )
                            draw_bounding_boxes(frame, class_idxs, result[2], result[1])

                            # Add the direction text to the image
                            cv2.putText(
                                frame,
                                f"HEAD ORIENTATION: {face_orientation.lower()}",
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                1,
                            )

                            if lip_distance > LIP_DISTANCE_THRESHOLD:
                                cv2.putText(
                                    frame,
                                    f"Mouth {face_idx} is Open !!!",
                                    (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    2,
                                )

                            cv2.putText(
                                frame,
                                f"Right Eye Position: {eye_position_right}",
                                (20, 400),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255),
                                1,
                            )

                            cv2.putText(
                                frame,
                                f"Left Eye Position: {eye_position_left}",
                                (20, 425),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255),
                                1,
                            )

                            draw_landmarks(frame, face_landmarks)
                        except Exception as e:
                            print(f"An error occurred - {e}")
                end_time = time.time()
                fps = int(1 / (end_time - start_time))

                if record:
                    out.write(frame)

                # Dashboard text
                # fps
                info5_text.write(
                    f"<h4 style='text-align:center;color:red;'>{int(fps)}</h4>",
                    unsafe_allow_html=True,
                )
                info1_text.write(
                    f"<h4 style='text-align:center;color:red;'>{str(face_count)}</h4>",
                    unsafe_allow_html=True,
                )
                info2_text.write(
                    f"<h4 style='text-align:center;color:red;'>{face_orientation}</h4>",
                    unsafe_allow_html=True,
                )
                info3_text.write(
                    f"<h4 style='text-align:center;color:red;'>{lip_status}</h4>",
                    unsafe_allow_html=True,
                )
                info4_text.write(
                    f"<p style='text-align:center;color:red;'><b>RIGHT:</b> {eye_position_right}, <b>LEFT:</b> {eye_position_left}</p>",
                    unsafe_allow_html=True,
                )

                # to display the image frame
                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=648, height=600)
                st_video_frame.image(frame, channels="BGR", use_column_width=True)

                if cv2.waitKey(1) == ord("q") or stop:
                    st_video_frame = st.empty()
                    vid = None
                    stop = None
                    break
        if vid:
            vid.release()
        cv2.destroyAllWindows()
