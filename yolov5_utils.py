import os
import cv2
import time
import numpy as np
from typing import Tuple, Union

ASSETS_DIR = os.path.join(os.path.dirname(__file__),"assets")
YOLO_ASSETS_DIR = os.path.join(ASSETS_DIR,"yolo")

COLORS = np.load(open(os.path.join(YOLO_ASSETS_DIR, "coco_colors.npy"), "rb"))

with open(os.path.join(YOLO_ASSETS_DIR, "classes.txt"), "r") as f:
    classes = [c.strip() for c in f.readlines()]


def detect_yolov5(image: np.ndarray, net) -> np.ndarray:
    """Compute detection predictions

    Args:
        image (np.ndarray): input image

    Returns:
        predictions (np.ndarray): computed detections
    """
    resized_img = cv2.resize(image, (640, 640), cv2.INTER_AREA)
    # normalize the image, swap Red and Blue channels, and generate the image blob
    blob = cv2.dnn.blobFromImage(
        resized_img, 1 / 255.0, (640, 640), swapRB=True, crop=False
    )
    # perform a forward pass to obtain the output predictions
    net.setInput(blob)
    start = time.time()
    predictions = net.forward()
    end = time.time()
    print(f"Total time elapsed for prediction: {(end-start):>.2f}s")
    output = predictions[0]
    return output


def process_detections(
    input_image: np.ndarray, output_predictions: np.ndarray, threshold: float = 0.45
) -> Tuple[list, list, list]:
    """Process the output predictions from the network

    Args:
        input_image (np.ndarray): The original input image
        output_predictions (np.ndarray): Output predictions from the network
        threshold (float): Confidence threshold for the predictions

    Returns:
        class_idxs (list): List of class IDs for the predicted classes
        confidences (list): List of confidences for the predicted classes
        bboxes (list): List of bounding boxes for the predicted classes
    """
    class_idxs = []
    confidences = []
    bboxes = []

    orig_h, orig_w = input_image.shape[:2]
    x_ratio = orig_w / 640
    y_ratio = orig_h / 640

    for i in range(output_predictions.shape[0]):
        row = output_predictions[i]
        confidence = row[4]
        # filter detections with confidence less than the specified threshold
        if confidence >= threshold:
            classes_scores = row[5:]
            # compute the class ID with the max score for each detection
            _, _, _, max_idx = cv2.minMaxLoc(classes_scores)
            class_id = max_idx[1]
            if classes_scores[class_id] > 0.25:
                confidences.append(confidence)
                class_idxs.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                # re-scale the xywh coordinates to the actual input dimensions
                left = int((x - 0.5 * w) * x_ratio)
                top = int((y - 0.5 * h) * y_ratio)
                width = int(w * x_ratio)
                height = int(h * y_ratio)
                box = np.array([left, top, width, height])
                bboxes.append(box)

    return class_idxs, confidences, bboxes


# NMS - To filter out overlapping detections
def apply_nms(
    class_idxs: Union[list, np.ndarray],
    bboxes: Union[list, np.ndarray],
    confidences: Union[list, np.ndarray],
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.4,
) -> Tuple[list, list, list]:
    """Apply Non-maximum suppression to filter out overlapping detections

    Args:
        class_idxs (list): The predicted class IDs from Yolo
        bboxes (list): The predicted bounding boxes from Yolo
        confidences (list): The predicted confidences from Yolo
        conf_threshold (float): The confidence threshold for the NMS, default = 0.5
        nms_threshold (float):

    Returns:
        result_class_idxs (list): Resulting class IDs after applying NMS
        result_confidences (list): Resulting confidences after applying NMS
        result_bboxes (list): Resulting bounding boxes after applying NMS
    """
    nms_idxs = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)
    result_class_idxs = []
    result_confidences = []
    result_bboxes = []

    for i in nms_idxs:
        result_class_idxs.append(class_idxs[i])
        result_confidences.append(confidences[i])
        result_bboxes.append(bboxes[i])

    return (result_class_idxs, result_confidences, result_bboxes)


# The full detection pipeline
def detect(
    input_image: np.ndarray,net, conf_threshold: float=0.45, nms_threshold: float=0.4
) -> Tuple[list, list, list]:
    output_predictions = detect_yolov5(input_image,net)
    class_idxs, confidences, bboxes = process_detections(
        input_image, output_predictions, threshold=conf_threshold
    )
    result_class_idxs, result_confidences, result_bboxes = apply_nms(
        class_idxs,
        bboxes,
        confidences,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
    return (result_class_idxs, result_confidences, result_bboxes)


# drawing the bouuding boxes on the image
def draw_bounding_boxes(
    image: np.ndarray,
    class_idxs: Union[list, np.ndarray],
    bboxes: Union[list, np.ndarray],
    confidences: Union[list, np.ndarray],
    show: bool = False,
) -> None:
    for i in range(len(class_idxs)):
        bbox = bboxes[i]
        class_id = class_idxs[i]
        color = COLORS[class_id]
        conf = confidences[i]
        cv2.rectangle(
            image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2
        )
        cv2.rectangle(
            image, (bbox[0], bbox[1] - 20), (bbox[0] + bbox[2], bbox[1]), color, -1
        )
        label = f"{classes[class_id]} {round(conf*100,2)}%"
        cv2.putText(
            image,
            label,
            (bbox[0] + 5, bbox[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )

    if show:
        cv2.imshow("frame", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    net = cv2.dnn.readNet(os.path.join(YOLO_ASSETS_DIR, "yolov5s.onnx"))
    test_img = cv2.imread(os.path.join(ASSETS_DIR, "Cars-on-highway.jpg"))
    result = detect(test_img,net)
    # print(result)
    image = test_img.copy()
    draw_bounding_boxes(image, result[0], result[2], result[1], show=True)
