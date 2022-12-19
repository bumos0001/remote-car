import requests
import time
import sys
import pwm_motor as motor
import threading
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse

# --------------Ubidots API 基本設定--------------------------------
TOKEN = "BBFF-eF9kEydp2aShBIlRzbNlKyhRoxNYC7"
DEVICE_KEY = '6396b9fc9689f4000daff2fa'    
URL = f"https://industrial.api.ubidots.com/api/v2.0/devices/{DEVICE_KEY}/variables/"
HEADERS = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}
auto_car_status = [0, 0, 0, 0]  # 自走車狀態 前後左右   index    0: 前  1:後   2:左    3:右

# ------------鏡頭------------------------------------------------
IM_WIDTH = 640  # Use smaller resolution for
IM_HEIGHT = 480  # slightly faster framerate

camera_type = 'usb'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
parser.add_argument('--picam', help='Use a picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'
if args.picam:
    camera_type = 'picamera'

sys.path.append('..')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX
# ---------------------------------------------------------------------------


# --------接收信號-----------------------------------------------
def get_var():
    while True:
        try:
            req = requests.get(url=URL, headers=HEADERS)
            data = req.json()['results']
            auto_car_status[3] = data[0]['lastValue']['value']
            auto_car_status[2] = data[1]['lastValue']['value']
            auto_car_status[1] = data[2]['lastValue']['value']
            auto_car_status[0] = data[3]['lastValue']['value']
        except Exception as e:
            print("[ERROR] Error posting, details: {}".format(e))


def open_camera():
    while True:
        if camera_type == 'usb':
            # Initialize USB webcam feed
            camera = cv2.VideoCapture(0, cv2.CAP_V4L)
            ret = camera.set(3, IM_WIDTH)
            ret = camera.set(4, IM_HEIGHT)

            while auto_car_status[1] == 1:
                ret, frame = camera.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_expanded = np.expand_dims(frame_rgb, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})

                # Draw the results of the detection (aka 'visulaize the results')
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    min_score_thresh=0.01)

                cs = np.squeeze(classes).astype(np.int32)
                sc = np.squeeze(scores)
                A = None
                for i in range(int(num[0])):
                    if cs[i] != 1 and sc[i] > 0.5:
                        cx = (boxes[0][i][1] + boxes[0][i][3]) / 2  # 找中心 x
                        cy = (boxes[0][i][0] + boxes[0][i][2]) / 2  # 找中心 y
                        A = np.round((boxes[0][i][3] - boxes[0][i][1]) * (boxes[0][i][2] - boxes[0][i][0]), 2)  # 算面積
                        frame = cv2.circle(frame, (int(cx * IM_WIDTH), int(cy * IM_HEIGHT)), radius=5,
                                           color=(0, 0, 255),
                                           thickness=-1)
                        if A > 0.4:  # 太近就響
                            motor.play_buzz()
                            print("太近")

                cv2.imshow('Object detector', frame)

                if auto_car_status[1] == 0:
                    break
            camera.release()
            cv2.destroyAllWindows()


get_status = threading.Thread(target=get_var)  # thread 接收遠端信號
get_status.start()
open_camera_ = threading.Thread(target=open_camera)  # thread 倒車顯影
open_camera_.start()

while True:
    time.sleep(0.1)
    print(auto_car_status)
    if auto_car_status[0] == 1:
        motor.forward()
        print("前進")
    if auto_car_status[1] == 1:
        motor.backward()
        print("後退")
    if auto_car_status[2] == 1:
        motor.turnLeft()
        print("左轉")
    if auto_car_status[3] == 1:
        motor.turnRight()
        print("右轉")
    if motor.print_distance() < 5:
        motor.play_buzz()
        print("太近")



