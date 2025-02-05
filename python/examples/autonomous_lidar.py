import sys
import os
import time

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from pathlib import Path

## adds the fsds package located the parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

from models.common import Conv
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box

import rospy
import tf
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from typing import List, Tuple, Dict
from datetime import datetime
from octanes_msgs.msg import ConeArray, Cone, ConePosition, ConeConfidence
import scipy.spatial
from utils.utils import DisjointSet

ConeType = Tuple[PointStamped, int, float]


class CameraPublisher:
    pubs_cam: Dict[str, List[rospy.Publisher]] = []

    camera_names = ["l", "r"]
    camera_images = ["rgb", "bb"]

    def __init__(self):
        rospy.init_node('simulation_yolov7')
        self.pub_cone_pos = rospy.Publisher('/sensorfusion/cone_pos', ConeArray, queue_size=10)
        self.transform_listener = tf.TransformListener()

    def transform_and_publish(self, cameras: List[List[Tuple[PointStamped, int, float]]]):
        cone_array: List[Cone] = []
        stamp = rospy.rostime.get_rostime()  # TODO: Average instead of taking last
        for camera_cones in cameras:
            if len(camera_cones) > 0:
                stamp = camera_cones[0][0].header.stamp

            for point, label, confidence in camera_cones:
                point_transformed = self.transform_listener.transformPoint("base_link", point).point
                x, y, z = point_transformed.x, point_transformed.y, point_transformed.z

                confidences = [0, 0, 0, 0]
                confidences[label] = confidence
                cone_array.append(
                    Cone(position=ConePosition(x, y), confidence=ConeConfidence(*confidences), type=label + 1))

        if len(cone_array) > 0:
            # Stretch points for distance calc to account for depth inaccuracy and l/r pos accuracy
            points = np.array([np.array([c.position.x * 0.4, c.position.y * 2.0]) for c in cone_array])
            tree = scipy.spatial.cKDTree(points)
            ds = DisjointSet()
            is_dup: Set[int] = set()
            for a, b in tree.query_pairs(0.5):
                ds.add(a, b)
                is_dup = is_dup.union([a, b])
            for group in ds.group.values():
                x = sum([cone_array[i].position.x for i in group]) / len(group)
                y = sum([cone_array[i].position.y for i in group]) / len(group)
                confidence = [sum([getattr(cone_array[i].confidence, c) for i in group]) / len(group) for c in
                              ConeConfidence.__slots__]
                cone_array.append(
                    Cone(position=ConePosition(x, y), confidence=ConeConfidence(*confidences),
                         type=np.argmax(confidence) + 1))
            for i in sorted(is_dup, reverse=True):
                del cone_array[i]
        if len(cone_array) >= 2:
            cone_array_msg = ConeArray(cones=cone_array)
            cone_array_msg.header.frame_id = 'base_link'
            cone_array_msg.header.stamp = stamp
            self.pub_cone_pos.publish(cone_array_msg)




# connect to the simulator 
client = fsds.FSDSClient()

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)

# Autonomous system constants
max_throttle = 0.2 # m/s^2
target_speed = 3 # m/s
max_steering = 0.3
cones_range_cutoff = 7 # meters

# YOLOvX Params.
weights = os.path.join("..", "..", "..", "yolo", "yolov7", "runs", "train", "yolov7-octanes7", "weights", "best.pt")
img_size = 416
device = 'cuda:0'
half = True
conf_thres = 0.25
iou_thres = 0.45



def pointgroup_to_cone(group):
    average_x = 0
    average_y = 0
    for point in group:
        average_x += point['x']
        average_y += point['y']
    average_x = average_x / len(group)
    average_y = average_y / len(group)
    return {'x': average_x, 'y': average_y}

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2))

def find_cones():
    # Get the pointcloud
    lidardata = client.getLidarData(lidar_name = 'Lidar_driving')

    # no points
    if len(lidardata.point_cloud) < 3:
        return []

    # Convert the list of floats into a list of xyz coordinates
    points = np.array(lidardata.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0]/3), 3))

    # Go through all the points and find nearby groups of points that are close together as those will probably be cones.

    current_group = []
    cones = []
    for i in range(1, len(points)):

        # Get the distance from current to previous point
        distance_to_last_point = distance(points[i][0], points[i][1], points[i-1][0], points[i-1][1])

        if distance_to_last_point < 0.1:
            # Points closer together then 10 cm are part of the same group
            current_group.append({'x': points[i][0], 'y': points[i][1]})
        else:
            # points further away indiate a split between groups
            if len(current_group) > 0:
                cone = pointgroup_to_cone(current_group)
                # calculate distance between lidar and cone
                if distance(0, 0, cone['x'], cone['y']) < cones_range_cutoff:
                    cones.append(cone)
                current_group = []
    return cones

def calculate_steering(cones):
    # If there are more cones on the left, go to the left, else go to the right.
    average_y = 0
    for cone in cones:
        average_y += cone['y']
    average_y = average_y / len(cones)

    if average_y > 0:
        return -max_steering
    else:
        return max_steering

def calculate_throttle():
    gps = client.getGpsData()

    # Calculate the velocity in the vehicle's frame
    velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    # the lower the velocity, the more throttle, up to max_throttle
    return (max_throttle * max(1 - velocity / target_speed, 0))
    
    
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output
    
    
def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        print(ckpt)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
        
        
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
    
    
def init_yolov7():
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    model.half()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1
    return model, stride, names, colors, old_img_w, old_img_h, old_img_b
    
    
def get_cone_distance(depth_img, xyxy, img_shape):
    idx = img_shape[1] * ((xyxy[3] + xyxy[1]) // 2) + ((xyxy[2] + xyxy[0]) // 2)
    return depth_img[idx]
    #depth_vals = []
    #for y in range(xyxy[3] - xyxy[1] + 1):
    #    base = img_shape[1] * (xyxy[1] + y)
    #    depth_vals.extend(depth_img[base + xyxy[0] : base + xyxy[2] + 1])
    #return min(depth_vals)
    
    
model, stride, names, colors, old_img_w, old_img_h, old_img_b = init_yolov7()   


def detect(im0, depth_img, frame): 
    global model, img_size, stride, device, conf_thres, iou_thres
    img = letterbox(im0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
        
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    
    cone_array: List[ConeType] = []
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            timestamp = rospy.Time.from_sec((datetime.now()).timestamp())

            # Write results
            for *xyxy, conf, cls in reversed(det):
                depth = get_cone_distance(depth_img, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), im0.shape)
                Y = -(depth * 1000) / 555.70372044077 * ((int(xyxy[2]) + int(xyxy[0])) / 2 - 450) / 1000
                X = math.sqrt(((depth * 1000) ** 2) - ((Y * 1000) ** 2) - (825 ** 2))
                label = f'{names[int(cls)]} {conf:.2f}'
                cone_array.append((PointStamped(
                                   Header(0, timestamp, "cam_" + frame + "_link"),
                                   Point(X / 1000.0, Y, 0)), int(cls), float(conf)))
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    #cv2.imwrite("coordinate.png", im0)
    
    # Stream results
    #cv2.imshow(frame, im0)
    #cv2.waitKey(1)  # 1 millisecond
    return cone_array
        


if __name__ == "__main__":
    publisher = CameraPublisher()
    while True:    
        left_rgb_request = fsds.ImageRequest(camera_name='cam_left_RGB', image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=True)
        right_rgb_request = fsds.ImageRequest(camera_name='cam_right_RGB', image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=True)
        left_depth_request = fsds.ImageRequest(camera_name='cam_left_depth', image_type=fsds.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
        right_depth_request = fsds.ImageRequest(camera_name='cam_right_depth', image_type=fsds.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
        img_left, img_right, img_left_depth, img_right_depth = client.simGetImages([left_rgb_request, right_rgb_request,\
                                                                                    left_depth_request, right_depth_request], vehicle_name = 'FSCar')
        img_left = cv2.imdecode(np.frombuffer(img_left.image_data_uint8, dtype="uint8"), cv2.IMREAD_COLOR)
        img_right = cv2.imdecode(np.frombuffer(img_right.image_data_uint8, dtype="uint8"), cv2.IMREAD_COLOR)
        
        cone_arrays: List[List[ConeType]] = []
        
        cone_arrays.append(detect(img_left, img_left_depth.image_data_float, "left"))
        cone_arrays.append(detect(img_right, img_right_depth.image_data_float, "right"))
        
        publisher.transform_and_publish(cone_arrays)

        cones = find_cones()
        if len(cones) != 0:
            car_controls = fsds.CarControls()
            car_controls.steering = calculate_steering(cones)
            car_controls.throttle = calculate_throttle()
            car_controls.brake = 0
            #client.setCarControls(car_controls)
