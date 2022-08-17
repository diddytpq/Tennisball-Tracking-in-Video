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
classes = [32]#[0, 38]
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


def outcome(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    tp = tn = fp1 = fp2 = fn = 0

    while i < n:
        if np.max(y_pred[i]) == 0 and np.max(y_true[i]) == 0:
            tn += 1
        elif np.max(y_pred[i]) > 0 and np.max(y_true[i]) == 0:
            fp2 += 1
        elif np.max(y_pred[i]) == 0 and np.max(y_true[i]) > 0:
            fn += 1
        elif np.max(y_pred[i]) > 0 and np.max(y_true[i]) > 0:
            #h_pred

            ball_cand_score = []

            y_pred_img = y_pred[i].copy()
            y_true_img = y_true[i].copy()


            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(y_pred_img, connectivity = 8)

            if len(stats): 
                stats = np.delete(stats, 0, axis = 0)
                centroids = np.delete(centroids, 0, axis = 0)

            for i in range(len(stats)):
                x, y, w, h, area = stats[i]

                score = np.mean(y_pred_img[y:y+h, x:x+w])

                ball_cand_score.append(score)

            (cx_pred, cy_pred) = centroids[np.argmax(ball_cand_score)]

            #h_true
            (cnts, _) = cv2.findContours(y_true_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for j in range(len(rects)):
                area = rects[j][2] * rects[j][3]
                if area > max_area:
                    max_area_idx = j
                    max_area = area
            target = rects[max_area_idx]
            (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

            #print((cx_pred, cy_pred))
            #print((cx_true, cy_true))
            dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))

            if dist > tol:
                fp1 += 1
            else:
                tp += 1
        i += 1
    return (tp, tn, fp1, fp2, fn)


def person_tracking(model, img, img_ori, device):

        person_box_left = []

        img_in = torch.from_numpy(img).to(device)
        img_in = img_in.float()
        img_in /= 255.0

        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)
        
        print(img_in.size())

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

                    person_box_left.append([x0, y0, x1, y1])
            
        return im0, person_box_left


def main(input_video):


    dT = 1 / 25

    cap_main = cv2.VideoCapture(input_video)

    fps = int(cap_main.get(cv2.CAP_PROP_FPS))

    if args.record:
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("test.mp4", codec, fps, (1280,720))

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
            person_tracking_img, person_box_left_list = person_tracking(model, img, img_ori, device)

        except:
            continue

        t2 = time.time()

        print("FPS : " , 1/(t2-t1))

        cv2.imshow('person_tracking_img',person_tracking_img)

        frame_record = person_tracking_img

        if args.record:
            out.write(frame_record)

        key = cv2.waitKey(0)

        if key == 27 : 
            cap_main.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":

    main(args.video_path)