#!/usr/bin/python
#-*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import torch.nn as nn
import time
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from PIL import Image
import os

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from face.mobilenet import MobileFacenet



def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(img,image_size,interpolation=cv2.INTER_AREA).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(img.permute(2, 0, 1).unsqueeze(0).float(),image_size).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize(image_size, Image.BILINEAR)
    return out


class Detector(nn.Module):
    def __init__(self,img,person_idx,imgsz,half,weight,use_l_r=False):
        super(Detector,self).__init__()

        self.query_feats=[]

        self.weights=weight
        self.half=half
        # set_logging()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())

        #人脸识别算法初始化
        self.face = MobileFacenet()

        ckpt = torch.load(r'C:\pycharm\人脸识别\MobileFaceNet_Pytorch-master\model\best\068.ckpt')
        self.face.load_state_dict(ckpt['net_state_dict'])
        self.face.to(self.device).eval()
        feat = self.face(img.to(self.device))
        #左右互换求均值
        if use_l_r:
            for i in range(0,len(feat),2):
                feats=feat[i:i+1,:]+feat[i+1:i+2,:]
                self.query_feats.append(feats)
            self.query_feats = torch.cat(self.query_feats, dim=0)  # torch.Size([N,512])
            self.query_feats = torch.nn.functional.normalize(self.query_feats, dim=1, p=2)
        else:
            self.query_feats = torch.nn.functional.normalize(feat, dim=1, p=2)
        self.person_idx = person_idx

    def forward(self,frame,conf_thres,iou_thres,face_thres=1.15):
        boxes=[]
        names = []
        all_images=[]
        gallery_loc = []

        new_img = [letterbox(frame, new_shape=640, auto=True)[0]]
        new_img = np.stack(new_img, 0)
        # Convert
        new_img = new_img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        new_img = np.ascontiguousarray(new_img)
        new_img = torch.from_numpy(new_img).to(self.device)
        new_img = new_img.half() if self.half else new_img.float()  # uint8 to fp16/32
        new_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if new_img.ndimension() == 3:
            new_img = new_img.unsqueeze(0)
        # Inference
        pred = self.model(new_img,augment=0)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,classes=81)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(new_img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # if (x2-x1)*(y2-y1)>=128*128:
                    gallery_loc.append([x1, y1, x2, y2])
                    box = [x1, y1, x2, y2]
                    face=crop_resize(frame,box,image_size=(96,112))
                    # img_cropped = F.to_tensor(np.float32(face))-127.5/128.0
                    img_cropped = F.to_tensor(np.float32(face))/ 255.0
                    all_images.append(img_cropped)
        if all_images:
            all_images=torch.stack(all_images,0).to(self.device)
            gallery_feats = self.face(all_images)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量
            m, n = self.query_feats.shape[0], gallery_feats.shape[0]
            # 计算人脸相似度
            distmat = torch.pow(self.query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) +torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(self.query_feats, gallery_feats.t(), beta=1, alpha=-2)
            for q_idx in range(m):
                idx = torch.argmin(distmat[q_idx])
                xy_distmat=distmat[q_idx][idx].item()
                if  xy_distmat<face_thres:
                    boxes.append(gallery_loc[idx])
                    names.append(self.person_idx[q_idx])
        return boxes, names


if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX
    reid_color = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (200, 0, 255), (255, 0, 200)]

    image_path=r'D:\AI_project\project\my_yolov5\face\images'
    queries=[]
    person_idx=[]
    for name in os.listdir(image_path):
        img=cv2.imread(os.path.join(image_path,name))
        # img=img[:,:,::-1]
        new_image=cv2.resize(img,(96,112),interpolation=cv2.INTER_AREA)
        lr_image=new_image[:,::-1,:]
        queries.append(F.to_tensor(np.float32(new_image)/255.0).unsqueeze(0))
        # queries.append(F.to_tensor(np.float32(lr_image) / 255.0).unsqueeze(0))
        person_idx.append(name.split('.')[0])
        # person_idx.append(name.split('.')[0]+'_001')
    queries = torch.cat(queries, dim=0)



    detect=Detector(queries,person_idx,640,True,r"C:\pycharm\yolov5\yolov5-2\runs\exp10\weights\best.pt")
    # path=r"C:\pycharm\人脸识别\facenet-pytorch-master\examples\video.mp4"
    # path=r'F:\reid测试视频\test1-2-slice0.avi'
    # cap = cv2.VideoCapture("rtsp://admin:admin12345@192.168.1.70:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1")
    cap = cv2.VideoCapture(0)
    count=0
    while True:
        ret, frame = cap.read()
        # cv2.namedWindow("Select target",0)
        if ret == False:
            break
        t=time.time()
        boxes, names = detect(frame, conf_thres=0.25, iou_thres=0.45)#frame[:,:,::-1]
        print('recognition Time:{0}'.format(time.time()-t))
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(frame, c1, c2, reid_color[i])
            cv2.putText(frame, names[i], (x1, y1 - 10), font, 1.2, reid_color[i], 2)
            cv2.imwrite("./result/{}.jpg".format(count),frame)
        cv2.imshow('video{}'.format(1), frame)
        cv2.waitKey(1)
        count+=1
    cv2.destroyWindow('video{}'.format(1))
    cv2.destroyWindow("Select target")