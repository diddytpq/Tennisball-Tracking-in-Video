#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.append(path + '/lib')  # add code to path


import numpy as np
import time
import cv2

import torch
import torch.backends.cudnn as cudnn

from tools import *

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

from yolov5.utils.augmentations import letterbox

from kalman_utils.KFilter import *

import argparse

parser = argparse.ArgumentParser(description = 'predict_tennis_ball_landing_point')

parser.add_argument('--video_path', type = str, default='videos/2.mov', help = 'input your video path')
parser.add_argument('--record', type = bool, default=False, help = 'set record video')
parser.add_argument('--debug', type = bool, default=False, help = 'set debug mod')


args = parser.parse_args()

device = 0
weights = path + "/lib/yolov5/weights/yolov5m6.pt"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
classes = None #[0, 38]
agnostic_nms = False
max_det = 1000
half=False
dnn = False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size


half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

if pt:
    model.model.half() if half else model.model.float()

cudnn.benchmark = True  # set True to speed up constant image size inference


color = tuple(np.random.randint(low=200, high = 255, size = 3).tolist())
color = tuple([0,125,255])

start_frame = 0

def person_tracking(model, img, img_ori, device):

        person_box_left = []
        person_box_right = []

        img_in = torch.from_numpy(img).to(device)
        img_in = img_in.float()
        img_in /= 255.0

        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)
        
        pred = model(img_in, augment=False, visualize=False)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # detections per image
            
            im0 = img_ori.copy()

            if len(det):
                det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class

                    label = names[c] #None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    x0, y0, x1, y1 = x0 - 10, y0 - 10, x1 + 10, y1 + 10

                    plot_one_box([x0, y0, x1, y1], im0, label=label, color=colors(c, True), line_thickness=3)

                    if y0 < (img_ori.shape[0] / 2) :
                        person_box_left.append([x0, y0, x1, y1])

                    else : 
                        person_box_right.append([x0, y0, x1, y1])
            
        return im0, person_box_left, person_box_right


def main(input_video):


    ball_esti_pos = []
    dT = 1 / 25

    cap_main = cv2.VideoCapture(input_video)

    fps = int(cap_main.get(cv2.CAP_PROP_FPS))

    if args.record:
        codec = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter("yolo_test_video.mp4", codec, fps, (1280,720))

    estimation_ball = Ball_Pos_Estimation()
    
    #칼만필터 초기 생성 후 삭제 --> 초기 생성한 칼만필터는 동작하지 않는 버그가 있음
    estimation_ball.kf = Kalman_filiter(0, 0, 0, dT)
    estimation_ball.kf.predict(dT)
    estimation_ball.reset_ball()


    disappear_cnt = 0
    court_img_reset_count = 0
    ball_pos_jrajectory = []

    total_frmae = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))

    print("total_frmae : ",total_frmae)

    cap_main.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #set start frame number

    while cap_main.isOpened():

        print("-----------------------------------------------------------------")
        t1 = time.time()

        frame_count = int(cap_main.get(cv2.CAP_PROP_POS_FRAMES))

        print("frame_count : ",frame_count)

        ret, frame = cap_main.read()

        # frame = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)

        # print(frame.shape)

        frame_mog2 = frame.copy()
        frame_yolo_main = frame.copy()

        img, img_ori = img_preprocessing(frame_yolo_main, imgsz, stride, pt)
        try:
            person_tracking_img, person_box_left_list, person_box_right_list = person_tracking(model, img, img_ori, device)

        except:
            continue



        t2 = time.time()

        print("FPS : " , 1/(t2-t1))

        cv2.imshow('person_tracking_img',person_tracking_img)


        if args.record:
            out.write(person_tracking_img)

        key = cv2.waitKey(1)

        if key == ord("c") or  court_img_reset_count >50 : 
            tennis_court_img = clear_tennis_court_img()
            court_img_reset_count = 0

        if key == 27 : 
            cap_main.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":

    main(args.video_path)