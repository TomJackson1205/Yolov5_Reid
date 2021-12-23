#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords,set_logging)
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg
from face.inception_resnet_v1 import InceptionResnetV1



def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(img,(image_size, image_size),interpolation=cv2.INTER_AREA).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(img.permute(2, 0, 1).unsqueeze(0).float(),(image_size, image_size)).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def creat_colos(target_name):
    color = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(target_name))]
    return dict(zip(target_name,color))


class Detector(nn.Module):
    def __init__(self, person_img=None, person_idx=None, use_Reid=False,imgsz=640, weight='weight/yolo_face.pt',
                    face_img=None,face_idx=None,  use_face_recognize=False, use_l_r=True, half=True):
        super(Detector,self).__init__()
        self.weights=weight
        self.half=half
        self.query_feats=[]
        self.use_face_recognize=use_face_recognize
        self.use_Reid=use_Reid
        ###################YOLOv5实例化，加载网络参数############################
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())
        if self.use_Reid:
            ###################Reid实例化，加载网络参数############################
            self.reidModel = build_model(reidCfg, num_classes=7868)
            self.reidModel.load_param(reidCfg.TEST.WEIGHT)
            self.reidModel.to(self.device).eval()
            ###################计算Query的特征向量############################
            feat = self.reidModel(person_img.to(self.device))
            self.query_feats.append(feat)
            self.query_feats = torch.cat(self.query_feats, dim=0)# torch.Size([2, 2048])
            self.query_feats = torch.nn.functional.normalize(self.query_feats, dim=1, p=2)
            self.person_idx=person_idx
            self.transfor=build_transforms(reidCfg)
        if self.use_face_recognize:
            self.face_query_feats = []
            ###################人脸识别算法初始化，加载网络参数############################
            self.face = InceptionResnetV1(pretrained='casia_webface').to(self.device).eval()
            ###################计算Query的特征向量############################
            face_feat = self.face(face_img.to(self.device))
            # 左右互换求均值
            self.face_query_feats=[]
            if use_l_r:
                for i in range(0, len(face_feat), 2):
                    face_feats = face_feat[i:i + 1, :] + face_feat[i + 1:i + 2, :]
                    self.face_query_feats.append(face_feats)
                self.face_query_feats = torch.cat(self.face_query_feats, dim=0)  # torch.Size([N,512])
            else:
                self.face_query_feats = torch.nn.functional.normalize(face_feat, dim=1, p=2)
            self.face_query_feats = torch.nn.functional.normalize(self.face_query_feats, dim=1, p=2)
            self.face_idx = face_idx



    def forward(self,frame,conf_thres,iou_thres,dist_thres=1.0,face_thres=1.0):
        boxes=[]
        names=[]
        gallery_img = []
        gallery_loc = []
        Reid_result = []

        face_boxes = []
        face_names = []
        face_gallery_img = []
        face_gallery_loc = []
        face_result=[]
        key = ('name', 'box')


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
        pred = non_max_suppression(pred, conf_thres, iou_thres,classes=[0,81])
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(new_img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    if self.use_Reid and cls == 0:
                        crop_img = frame[ymin:ymax, xmin:xmax]
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                        crop_img = self.transfor(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                        gallery_img.append(crop_img)
                        gallery_loc.append((xmin, ymin, xmax, ymax, cls.item()))
                    elif self.use_face_recognize and cls == 81:
                        face_box = [xmin, ymin, xmax, ymax]
                        face = crop_resize(frame[:,:,::-1], face_box, image_size=112)
                        img_cropped = F.to_tensor(np.float32(face)) / 255.0
                        face_gallery_img.append(img_cropped)
                        face_gallery_loc.append((xmin, ymin, xmax, ymax, cls.item()))
                    else:
                        boxes.append([xmin, ymin, xmax, ymax,cls.item()])
        #Reid
        if self.use_Reid and gallery_img:
            gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
            gallery_img = gallery_img.to(self.device)
            gallery_feats = self.reidModel(gallery_img)  # torch.Size([7, 2048])
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量
            m, n = self.query_feats.shape[0], gallery_feats.shape[0]
            #计算相似度（Reid独有的算法）
            distmat = torch.pow(self.query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(self.query_feats, gallery_feats.t(),beta=1, alpha=-2)
            for q_idx in range(m):
                idx=torch.argmin(distmat[q_idx]).item()
                if distmat[q_idx][idx].item()<dist_thres:
                    Reid_result.append(dict(zip(key, [self.person_idx[q_idx], gallery_loc[idx]])))
        # 人脸识别
        if self.use_face_recognize and face_gallery_img:
            all_images=torch.stack(face_gallery_img,0).to(self.device)
            face_gallery_feats = self.face(all_images)
            face_gallery_feats = torch.nn.functional.normalize(face_gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量
            m, n = self.face_query_feats.shape[0], face_gallery_feats.shape[0]
            # 计算人脸相似度
            face_distmat = torch.pow(self.face_query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) +\
                           torch.pow(face_gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            face_distmat.addmm_(self.face_query_feats, face_gallery_feats.t(), beta=1, alpha=-2)
            for q_idx in range(m):
                idx = torch.argmin(face_distmat[q_idx])
                xy_distmat = face_distmat[q_idx][idx].item()
                if xy_distmat < face_thres:
                    face_result.append(dict(zip(key, [self.face_idx[q_idx], face_gallery_loc[idx]])))
                    # face_boxes.append(face_gallery_loc[idx])
                    # face_names.append(self.face_idx[q_idx])
        if self.use_face_recognize and self.use_Reid:
            return Reid_result,face_result
        elif self.use_Reid:
            return Reid_result
        elif self.use_face_recognize:
            return face_result
        else:
            return boxes


if __name__ == '__main__':

    target_sum=1

    use_Reid=True
    dist_thres=1.0
    queries = []
    person_idx = []
    reid_query_path=r"query"

    use_face_recognize=True
    face_thres=1.0
    use_l_r=True
    face_queries = []
    face_idx = []
    face_query_path = r"face_query"

    if use_face_recognize:
        weight_path = 'weight/face_detect/face_L.pt'
    else:
        weight_path = 'weight/yolov5l.pt'


    if use_Reid:
        #Reid
        for name in os.listdir(reid_query_path):
            query=cv2.imread(os.path.join(reid_query_path,name))
            query = Image.fromarray(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
            query = build_transforms(reidCfg)(query).unsqueeze(0)
            queries.append(query)
            person_idx.append(name.split('.')[0])
        queries=torch.cat(queries, dim=0)
    colors = creat_colos(person_idx)
    if use_face_recognize:
        # 人脸识别
        for name in os.listdir(face_query_path):
            img = cv2.imread(os.path.join(face_query_path, name))
            img = img[:, :, ::-1]
            new_image = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            lr_image = new_image[:, ::-1, :]
            face_queries.append(F.to_tensor(np.float32(new_image) / 255.0).unsqueeze(0))
            face_queries.append(F.to_tensor(np.float32(lr_image) / 255.0).unsqueeze(0))
            face_idx.append(name.split('.')[0])
        face_queries = torch.cat(face_queries, dim=0)
        # colors=creat_colos(face_idx)

    ''' person_img = None,person_idx = None,use_Reid = True, 
        imgsz = 640, weight = 'weight/yolo_face.pt',
        face_img = None, face_idx = None, use_face_recognize = False, use_l_r = True,
        half = True'''
    if use_face_recognize and use_Reid:
        detect = Detector(person_img=queries, person_idx=person_idx, use_Reid=True, imgsz=640, weight=weight_path,
                          face_img=face_queries, face_idx=face_idx, use_face_recognize=use_face_recognize, use_l_r=use_l_r)
    elif use_Reid:
        detect=Detector(person_img=queries, person_idx=person_idx, use_Reid=True, imgsz=640, weight=weight_path)
    elif use_face_recognize:
        detect = Detector(face_img=face_queries, face_idx=face_idx, use_face_recognize=use_face_recognize, use_l_r=use_l_r, imgsz=640, weight=weight_path)
    else:
        detect = Detector(imgsz=640, half=False, weight=weight_path)

    cap = cv2.VideoCapture(r"test_video/4c8b26784bd33fb914e73ae9cf68cd0e1619761655.mp4")
    cap1 = cv2.VideoCapture(r"test_video/20210427143527-20210427143727_1.mp4")
    cap2 = cv2.VideoCapture(r"test_video/20210427143624-20210427143824_2.mp4")
    cap3 = cv2.VideoCapture(r"test_video/20210427143730-20210427144054_1.mp4")
    cap4 = cv2.VideoCapture(r"test_video/20210427143800-20210427144000_1.mp4")
    cap5 = cv2.VideoCapture(r"test_video/20210427143919-20210427144119_1.mp4")

    width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out.avi', fourcc, 25.0, (3*width, 2*height))
    while True:
        ret, fram = cap.read()
        ret1, fram1 = cap1.read()
        ret2, fram2 = cap2.read()
        ret3, fram3 = cap3.read()
        ret4, fram4 = cap4.read()
        ret5, fram5 = cap5.read()


        font = cv2.FONT_HERSHEY_SIMPLEX
        if ret == False and ret1==False and ret2==False and ret3==False and ret4==False and ret5==False:
            break
        if ret:
            Reid_result,face_result= detect(fram, conf_thres=0.25, iou_thres=0.45, dist_thres=2.0, face_thres=1.0)
            print('frame',Reid_result)
            print('frame',face_result)
            for name_and_box in face_result:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
            for name_and_box in Reid_result:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
        else:
            fram=(np.ones((height, width, 3))* 128).astype(np.uint8)

        if ret1:
            Reid_result1,face_result1= detect(fram1, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres, face_thres=face_thres)
            print('frame1',Reid_result1)
            for name_and_box in Reid_result1:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram1, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram1, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
        else:
            fram1 = (np.ones((height, width, 3))* 128).astype(np.uint8)

        if ret2:
            Reid_result2,face_result3= detect(fram2, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres, face_thres=face_thres)
            print('frame2',Reid_result2)
            for name_and_box in Reid_result2:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram2, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram2, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
        else:
            fram2 = (np.ones((height, width, 3))* 128).astype(np.uint8)

        if ret3:
            Reid_result3,face_result3= detect(fram3, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres, face_thres=face_thres)
            print('frame3',Reid_result3)
            for name_and_box in Reid_result3:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram3, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram3, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
        else:
            fram3 = (np.ones((height, width, 3))* 128).astype(np.uint8)
        if ret4:
            Reid_result4,face_result4= detect(fram4, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres, face_thres=face_thres)
            print('frame4',Reid_result4)
            for name_and_box in Reid_result4:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram4, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram4, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
        else:
            fram4 = (np.ones((height, width, 3))* 128).astype(np.uint8)
        if ret5:
            Reid_result5,face_result5 = detect(fram5, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres,face_thres=face_thres)
            print('frame5:',Reid_result5)
            for name_and_box in Reid_result5:
                x1, y1, x2, y2, cls = name_and_box['box']
                c1, c2 = (x1, y1), (x2, y2)
                cv2.rectangle(fram5, c1, c2, colors[name_and_box['name']])
                cv2.putText(fram5, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
        else:
            fram5 = (np.ones((height, width, 3))* 128).astype(np.uint8)


        # 同时显示6个摄像头
        frameLeftUp = cv2.resize(fram, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameCenterUp = cv2.resize(fram1, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = cv2.resize(fram2, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(fram3, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameCenterDown = cv2.resize(fram4, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameRightDown = cv2.resize(fram5, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameUp = np.hstack((frameLeftUp, frameCenterUp, frameRightUp))
        frameDown = np.hstack((frameLeftDown, frameCenterDown, frameRightDown))

        frame = np.vstack((frameUp, frameDown))
        out.write(frame)
        # cv2.imshow("target", frame)
        # cv2.waitKey(1)
    # cv2.destroyAllWindows()


 # cap = cv2.VideoCapture(r"/media/xianyu/ESD-USB/video/广严大道忠孝楼后大门_20210427143730-20210427144054_1.mp4")
    # cap = cv2.VideoCapture(r"/media/xianyu/ESD-USB/video/至善西路路口_20210427143919-20210427144119_1.mp4")
    # width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    # height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/home/xianyu/PycharmProjects/Reid/video/至善西路路口_20210427143919-20210427144119_1.mp4', fourcc, 25.0, (2 * width, 2 * height))
    # while True:
    #     ret, fram = cap.read()
    #     # cv2.namedWindow('target',0)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     if ret == False:
    #         break
    #     if use_face_recognize and use_Reid:
    #         boxes, names, face_boxes, face_names= detect(fram, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres, face_thres=face_thres)
    #     elif use_Reid:
    #         results = detect(fram, conf_thres=0.25, iou_thres=0.45, dist_thres=dist_thres)
    #     elif use_face_recognize:
    #         results = detect(fram, conf_thres=0.25, iou_thres=0.45, face_thres=face_thres)
    #     else:
    #         boxes = detect(fram, conf_thres=0.25, iou_thres=0.45)
    #
    #     print(results)
    #     for name_and_box in results:
    #         x1, y1, x2, y2, cls = name_and_box['box']
    #         c1, c2 = (x1, y1), (x2, y2)
    #         cv2.rectangle(fram, c1, c2, colors[name_and_box['name']])
    #         cv2.putText(fram, name_and_box['name'], (x1, y1 - 10), font, 1.2, colors[name_and_box['name']], 2)
    #     out.write(fram)
    #     cv2.imshow("target", fram)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()
