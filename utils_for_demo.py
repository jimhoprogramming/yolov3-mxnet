# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import mxnet as mx
import numpy as np
from mxnet import nd, gluon
import threading
import xml.etree.ElementTree as ET
import re
import pandas as pd
import sys, os
sys.path.append('../dress')
from load_data import show_jpg_result

def try_gpu(num_list):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    ctx = []
    for num in num_list:
        try:
            tmp_ctx = mx.gpu(int(num))
            _ = nd.array([0], ctx=tmp_ctx)
            ctx.append(tmp_ctx)
        except Exception as e:
            print("gpu {}:".format(num), e)
    if not ctx:
        ctx.append(mx.cpu())
    return ctx


def bbox_iou(box1, box2, transform=True,ctx = None):
    '''
        判断预测盒子和实际盒子的重合度。>0.5是比较好的预测
    '''
    
    ctx = ctx
    if not isinstance(box1, nd.NDArray):
        box1 = nd.array(box1, ctx = ctx)
    if not isinstance(box2, nd.NDArray):
        box2 = nd.array(box2, ctx = ctx)
    box1 = nd.abs(box1)
    box2 = nd.abs(box2)

    if transform:
        tmp_box1 = box1.copy()
        tmp_box1[:, 0] = box1[:, 0] - box1[:, 2] / 2.0
        tmp_box1[:, 1] = box1[:, 1] - box1[:, 3] / 2.0
        tmp_box1[:, 2] = box1[:, 0] + box1[:, 2] / 2.0
        tmp_box1[:, 3] = box1[:, 1] + box1[:, 3] / 2.0
        box1 = tmp_box1
        tmp_box2 = box2.copy()
        tmp_box2[:, 0] = box2[:, 0] - box2[:, 2] / 2.0
        tmp_box2[:, 1] = box2[:, 1] - box2[:, 3] / 2.0
        tmp_box2[:, 2] = box2[:, 0] + box2[:, 2] / 2.0
        tmp_box2[:, 3] = box2[:, 1] + box2[:, 3] / 2.0
        box2 = tmp_box2
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = nd.where(b1_x1 > b2_x1, b1_x1, b2_x1)
    inter_rect_y1 = nd.where(b1_y1 > b2_y1, b1_y1, b2_y1)
    inter_rect_x2 = nd.where(b1_x2 < b2_x2, b1_x2, b2_x2)
    inter_rect_y2 = nd.where(b1_y2 < b2_y2, b1_y2, b2_y2)

    # Intersection area
    inter_area = nd.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=10000) * nd.clip(
        inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=10000)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    # iou[inter_area >= b1_area] = 0.8
    # iou[inter_area >= b2_area] = 0.8
    return nd.clip(iou, 1e-5, 1. - 1e-5)


def predict_transform(prediction, input_dim, anchors):
    '''
        功能：
        输入：
            prediction:经过神经网络，上下采样的数量总和x3的x,y,w,h,pc,c1,c2的原始值[batchnumber,13x13x3+26x26x3+52x52x3,7]
            input_dim:416
            anchors:九个锚框尺寸对
        输出：
            prediction：所有锚框实际值[batchnumber,13x13x3+26x26x3+52x52x3,7]
    '''
    ctx = prediction.context
    b_xywhs = prediction.copy()
    if not isinstance(anchors, nd.NDArray):
        anchors = nd.array(anchors, ctx=ctx)
    #print('sum(prediction[:,4]==1):{}'.format(nd.sum(prediction[:,:,4]==1)))
    batch_size = prediction.shape[0]
    anchors_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    strides = [13, 26, 52]
    step = [(0, 507), (507, 2535), (2535, 10647)]
    for i in range(3):
        stride = strides[i]
        grid = np.arange(stride)
        a, b = np.meshgrid(grid, grid)
        x_offset = nd.array(a.reshape((-1, 1)), ctx=ctx)
        y_offset = nd.array(b.reshape((-1, 1)), ctx=ctx)
        x_y_offset = \
            nd.repeat(
                nd.expand_dims(
                    nd.repeat(
                        nd.concat(
                            x_offset, y_offset, dim=1), repeats=3, axis=0
                    ).reshape((-1, 2)),
                    0
                ),
                repeats=batch_size, axis=0
            )
        tmp_anchors = \
            nd.repeat(
                nd.expand_dims(
                    nd.repeat(
                        nd.expand_dims(
                            anchors[anchors_masks[i]], 0
                        ),
                        repeats=stride * stride, axis=0
                    ).reshape((-1, 2)),
                    0
                ),
                repeats=batch_size, axis=0
            )

        prediction[:, step[i][0]:step[i][1], :2] += x_y_offset
        prediction[:, step[i][0]:step[i][1], :2] *= (float(input_dim) * 1.0/ stride)
        prediction[:, step[i][0]:step[i][1], 2:4] = \
            nd.exp(prediction[:, step[i][0]:step[i][1], 2:4]) * tmp_anchors
    #print('model predict_transform sum(prediction[:,4]==1):{}'.format(nd.sum(prediction[:,:,4]==1)))
    return prediction


def write_results(prediction, num_classes, confidence=0.5, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).expand_dims(2)
    prediction = prediction * conf_mask

    batch_size = prediction.shape[0]

    box_corner = nd.zeros(prediction.shape, dtype="float32")
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = None

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf = nd.max(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf_score = nd.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf = max_conf.astype("float32").expand_dims(1)
        max_conf_score = max_conf_score.astype("float32").expand_dims(1)
        image_pred = nd.concat(image_pred[:, :5], max_conf, max_conf_score, dim=1).asnumpy()
        non_zero_ind = np.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_ind, :].reshape((-1, 7))
        except Exception as e:
            print(e)
            continue
        if image_pred_.shape[0] == 0:
            continue
        # Get the various classes detected in the image
        img_classes = np.unique(image_pred_[:, -1])
        # -1 index holds the class index

        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * np.expand_dims(image_pred_[:, -1] == cls, axis=1)
            class_mask_ind = np.nonzero(cls_mask[:, -2])
            image_pred_class = image_pred_[class_mask_ind].reshape((-1, 7))

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = np.argsort(image_pred_class[:, 4])[::-1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.shape[0]

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    box1 = np.expand_dims(image_pred_class[i], 0)
                    box2 = image_pred_class[i + 1:]
                    if len(box2) == 0:
                        break
                    box1 = np.repeat(box1, repeats=box2.shape[0], axis=0)
                    ious = bbox_iou(box1, box2, transform=False).asnumpy()
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = np.expand_dims(ious < nms_conf, 1).astype(np.float32)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = np.nonzero(image_pred_class[:, 4])
                image_pred_class = image_pred_class[non_zero_ind].reshape((-1, 7))

            batch_ind = np.ones((image_pred_class.shape[0], 1)) * ind

            seq = nd.concat(nd.array(batch_ind), nd.array(image_pred_class), dim=1)

            if output is None:
                output = seq
            else:
                output = nd.concat(output, seq, dim=0)
    return output


def letterbox_image_jim(img, inp_dim, labels=None):
    '''
        把原图片生成指定尺寸的画布大小，同时缩放标签中x,y,w,h
    '''    
    img_h, img_w = img.shape[0], img.shape[1]     # 行代表高 ， 列代表宽

    
    w, h = inp_dim
    scale = min(w * 1.0/img_w, h * 1.0/img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    dx = (w-new_w) // 2
    dy = (h-new_h) // 2
    #print(u'比率:{}'.format(scale))
    #print(u'原图尺寸：{}'.format(img.shape))
    #print(u'新尺寸：{}'.format((new_h,new_w)))
    
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)   # 改变图像尺寸[fx，fy]
##    print(u'缩放后真正尺寸：{}'.format(resized_image.shape))
    if labels is not None:
##        print(u'步进修改labels')
        mask = labels > 0.
        labels[:, 0] = labels[:, 0] * scale + dx 
        labels[:, 1] = labels[:, 1] * scale + dy 
        labels[:, 2] = labels[:, 2] * scale
        labels[:, 3] = labels[:, 3] * scale
        labels *= mask
    canvas = np.full((h, w, 3), 128, dtype=np.uint8)

    canvas[dy : dy + new_h, dx : dx + new_w, :] = resized_image

    return canvas, labels

def letterbox_image(img, inp_dim, labels=None):
    '''
        把原图片生成指定尺寸的画布大小，同时缩放标签中x,y,w,h
    '''    
    img_h, img_w = img.shape[0], img.shape[1]     # 行代表高 ， 列代表宽

    
    w, h = inp_dim
    scale = min(w * 1.0/img_w, h * 1.0/img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    dx = (w-new_w) // 2
    dy = (h-new_h) // 2
    #print(u'比率:{}'.format(scale))
    #print(u'原图尺寸：{}'.format(img.shape))
    #print(u'新尺寸：{}'.format((new_h,new_w)))
    
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)   # 改变图像尺寸[fx，fy]
##    print(u'缩放后真正尺寸：{}'.format(resized_image.shape))
    if labels is not None:
##        print(u'步进修改labels')
        mask = labels > 0.
        labels[:, 0] = (labels[:, 0] * new_w + dx) / w 
        labels[:, 1] = (labels[:, 1] * new_h + dy) / h 
        labels[:, 2] = labels[:, 2] * new_w / w
        labels[:, 3] = labels[:, 3] * new_h / h 
        labels *= mask
    canvas = np.full((h, w, 3), 128, dtype=np.uint8)

    canvas[dy : dy + new_h, dx : dx + new_w, :] = resized_image

    return canvas, labels


def prep_image(img, inp_dim, labels=None):
    
    img, labels = letterbox_image(img, (inp_dim, inp_dim), labels)
    img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype("float32")
    img /= 255.0
    if labels is not None:
        return img, labels
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.readlines()
    names = [name.strip() for name in names]
    if names[-1] == "\n":
        names.pop(-1)
    return names


def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]

def unsplit_and_load(data, ctx):
    n, k = data.shape[0], 1
    m = n // k
    return [data[i * m: (i + 1) * m].as_in_context(ctx) for i in range(k)]

class SigmoidBinaryCrossEntropyLoss(gluon.loss.Loss):
    def __init__(self, from_sigmoid=False, weight=1, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        if not self._from_sigmoid:
            # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
            tmp_loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            tmp_loss = -(F.log(pred + 1e-12) * label + F.log(1. - pred + 1e-12) * (1. - label))
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss


class L1Loss(gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        tmp_loss = F.abs(pred - label)
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss


class L2Loss(gluon.loss.Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gluon.loss._reshape_like(F, label, pred)
        tmp_loss = F.square(pred - label)
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight / 2, sample_weight)
        return tmp_loss


class FocalLoss(gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, gamma=2, eps=1e-7, alpha=0.25, with_ce=False):
        super(FocalLoss, self).__init__(weight=weight, batch_axis=batch_axis)
        self.gamma = gamma
        self.eps = eps
        self.with_ce = with_ce
        self.alpha = alpha

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if self.alpha > 0.:
            alpha_t = F.abs(self.alpha + (label == 1.).astype("float32") - 1.)
        else:
            alpha_t = 1.
        if self.with_ce:
            sce_loss = SigmoidBinaryCrossEntropyLoss()
            p_t = sce_loss(pred, label)
            tmp_loss = -(alpha_t * F.power(1 - p_t, self.gamma) * p_t)
        else:
            p_t = F.clip(F.abs(pred + label - 1.), a_min=self.eps, a_max=1. - self.eps)
            tmp_loss = -(alpha_t * F.power(1 - p_t, self.gamma) * F.log(p_t))
        tmp_loss = gluon.loss._apply_weighting(F, tmp_loss, self._weight, sample_weight)
        return tmp_loss


class HuberLoss(gluon.loss.Loss):
    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label =gluon.loss. _reshape_like(F, label, pred)
        loss = F.clip(F.abs(pred - label), a_min=1e-7, a_max=10000.)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho,
                       (0.5/self._rho) * F.power(loss, 2))
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        return loss


class LossRecorder(mx.metric.EvalMetric):
    """LossRecorder is used to record raw loss so we can observe loss directly
    """

    def __init__(self, name):
        super(LossRecorder, self).__init__(name)

    def update(self, labels, preds=0):
        """Update metric with pure loss
        """
        for loss in labels:
            self.sum_metric += np.mean(loss.copy().asnumpy())
            self.num_inst += 1


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
            # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception as e:
            print(e)
            return None


def parse_xml(xml_file, classes):
    root = ET.parse(xml_file).getroot()
    image_size = root.find("size")
    size = {
        "width": float(image_size.find("width").text),
        "height": float(image_size.find("height").text),
        "depth": float(image_size.find("depth").text)
    }
    bbox = []
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes)
    for obj in root.findall("object"):
        cls = np.argwhere(classes == obj.find("name").text).reshape(-1)[0]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        center_x = (xmin + xmax) / 2.0 / size["width"]
        center_y = (ymin + ymax) / 2.0 / size["height"]
        width = (xmax - xmin) / size["width"]
        height = (ymax - ymin) / size["height"]
        bbox.append([cls, center_x, center_y, width, height])
    return np.array(bbox)


def prep_label(label_file, classes):
    ''' 
        功能：生成一个张量[x,y,w,h,pc,c0,.....,c24],其中pc = 1时含有效数据。0时提醒忽略该行
        输入：[c,x,y,w,h] X m  ,一幅图多行表示含有多个类别物体存在 
        输出：shape = [30,29] 张量
    '''
    num_classes = len(classes)
    if isinstance(label_file, list):
        labels = label_file
    elif label_file.endswith(".txt"):
        with open(label_file, "r") as file:
            labels = file.readlines()
            labels = np.array([list(map(float, x.split())) for x in labels], dtype="float32")  # debuging 2018-11-27
##        labels = np.array(pd.read_csv(label_file),dtype='float32')
    elif label_file.endswith(".xml"):
        labels = parse_xml(label_file, classes)
    final_labels = nd.zeros(shape=(30, num_classes + 5), dtype="float32")      
    i = 0
    for label in labels:
        one_hot = np.zeros(shape=(num_classes + 5), dtype="float32")
        one_hot[5 + int(label[0])] = 1.0
        one_hot[4] = 1.0
        one_hot[:4] = label[1:]
        final_labels[i] = one_hot
        i += 1
        i %= 30
    return nd.array(final_labels)


def prep_final_label(labels, num_classes, input_dim=416):
    '''
        输入：
            labels : 416尺寸变形后的结果集标签[30行,[x,y,w,h,pc,c0-c23]=5+24=29]
            num_classes : 数值=24类
            imput_dim : 图像输入卷积的统一尺寸
        输出：
            t_y：[8,10647,7]=[batch_num,13x13x3+26x26x3+52x52x3,[tx,ty,tw,th,pc,c1,c2]
            t_xywhs：[8,10647,5]=[batch_num,13x13x3+26x26x3+52x52x3,[x,y,w,h,pc]
            
            
    '''
    ctx = labels.context
    # define 9 boxs
    anchors = nd.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)], ctx=ctx)
    # define 3 group 
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # define 3 classes size box for label = (size, size, 锚框数=3对, 29) 
    label_1 = nd.zeros(shape=(13, 13, 3, num_classes + 5), dtype="float32", ctx=ctx)
    label_2 = nd.zeros(shape=(26, 26, 3, num_classes + 5), dtype="float32", ctx=ctx)
    label_3 = nd.zeros(shape=(52, 52, 3, num_classes + 5), dtype="float32", ctx=ctx)

    # define 3 classes size box for true label = (size, size, 3, 5)
    true_label_1 = nd.zeros(shape=(13, 13, 3, 5), dtype="float32", ctx=ctx)
    true_label_2 = nd.zeros(shape=(26, 26, 3, 5), dtype="float32", ctx=ctx)
    true_label_3 = nd.zeros(shape=(52, 52, 3, 5), dtype="float32", ctx=ctx)

    # define list save 3 different size box label and true label
    label_list = [label_1, label_2, label_3]
    true_label_list = [true_label_1, true_label_2, true_label_3]

    # 逐行处理[x,y,w,h,pc,c,.....,c24]
    m, n = labels.shape
    for x_box in range(m):
##        print(u'正在处理第{}个对象'.format(x_box))
        if labels[x_box, 4].asscalar() == 0.0:
##            print(u'step labels[{},4]==0.0,退出一幅图片处理。'.format(x_box))
            break
        
        # 循环得到13,26,52步长 = 三个单元框的大小 13x13  26x26 52x52
        for i in range(3):        
            stride = 2 ** i * 13
            
            tmp_anchors = anchors[anchors_mask[i]]  # 得到一组含3个锚框尺寸   [3,2]

            # 装实y[3,[x,y,w,h]*13]
            tmp_xywh = nd.repeat(nd.expand_dims(labels[x_box, :4] * stride, axis=0),
                                 repeats=tmp_anchors.shape[0], axis=0)     # [3, 4]每行代表相同锚框x3，列是[x, y, w, h]*单元框宽                                                          

            # 装[ix3 , [x, y, 锚框i_w*13/416, 锚框i_h*13/416]]
            anchor_xywh = tmp_xywh.copy()
            anchor_xywh[:, 2:4] = tmp_anchors / input_dim * stride         # [3, 4]

            # 得到最接近的锚框序号
            best_anchor = nd.argmax(bbox_iou(tmp_xywh, anchor_xywh), axis=0)
##            print(u'选最好的预测大小[13,26,52]框的序号是：{}'.format(best_anchor))

            # 计算盒子的位置索引
            label = labels[x_box].copy()   # 已经调整尺寸的y shape = [1,29]  行=[x,y,w,h,pc,c0,c1,......,c24]
            k = nd.floor(label[:2] * stride)
            label[:2] =  label[:2] * stride - k   # [x,y]*13 - [x,y]*13取整=余数
##            print(u'索引序号变化结果：{}'.format(label[:2]))
            
            tmp_idx = k  # [x,y]*13 四写五入=取整
            tmp_idx = tmp_idx.astype("int")
##            print(u'临时索引的值：{}'.format(tmp_idx))
            label[2:4] = nd.log(label[2:4] * input_dim / tmp_anchors[best_anchor].reshape(-1) + 1e-12)

            true_xywhs = labels[x_box, :5] * input_dim
            true_xywhs[4] = 1.0                

            label_list[i][tmp_idx[1], tmp_idx[0], best_anchor] = label    # here
##            print('sum(true_xywhs[4]==1):{}'.format(nd.sum(true_xywhs[4]==1)))
            true_label_list[i][tmp_idx[1], tmp_idx[0], best_anchor] = true_xywhs

    t_y = nd.concat(label_list[0].reshape((-1, num_classes + 5)),
                    label_list[1].reshape((-1, num_classes + 5)),
                    label_list[2].reshape((-1, num_classes + 5)),
                    dim=0)
    
    t_xywhs = nd.concat(true_label_list[0].reshape((-1, 5)),
                        true_label_list[1].reshape((-1, 5)),
                        true_label_list[2].reshape((-1, 5)),
                        dim=0)
    return t_y, t_xywhs

##def calculate_ignore(prediction, true_xywhs, ignore_thresh=0.5):
##    '''
##        功能：计算被忽略的边界框是哪些
##        输入：
##            prediction:被预测的结果[8，10647,4+1+cls_num]=[8,10647,7]
##            true_xywhs:实际标签的转换值[8，10647,4+1+cls_num]=[8,10647,5]
##            ignore_thresh:百分比阀值0.5
##        通过全局变量透传的参数：
##                        输入维度416
##                        9个表框尺寸列表
##                        预测是否含有对象的结果张量[8=batch_num,10647,1]
##                        
##        输出:
##            ignore_mask:[batch_num,10647,1]实际标签含有物品且预测位置不重合的边框中心所有锚框=1
##    '''
##    print(prediction.shape)
##    print(true_xywhs.shape)
##    ctx = prediction.context
##    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
##                        (59, 119), (116, 90), (156, 198), (373, 326)])    
##    tmp_pred = predict_transform(prediction, input_dim=416,anchors=anchors)  # 将预测结果转换成需要的标签形式，关键！[8,10647,7]
##    
##    tmp_pred_numpy = tmp_pred.asnumpy()
##    index = np.where(tmp_pred_numpy[:,:,4]==1)
##    print('model ignore index:\n{}'.format(index))
##    tmp_pred_numpy = tmp_pred_numpy[:,index[1],:]
##    print('tmp_pred_numpy:\n{}'.format(tmp_pred_numpy))
##    
##    # 生成一个带批次有否物品的1值张量[8,10647,1]
##    ignore_mask = nd.ones(shape=(prediction.shape[0],prediction.shape[1],1), dtype="float32", ctx = ctx)
##    item_index = np.argwhere(true_xywhs.asnumpy()[:, :, 4] == 1.0) # 实际标签含物品才有效=索引号的列表
##    
##    for x_box, y_box in item_index:
##        print(tmp_pred[x_box, y_box:y_box + 1, :5])
##        print(true_xywhs[x_box, y_box:y_box + 1])
##        iou = bbox_iou(tmp_pred[x_box, y_box:y_box + 1, :5], true_xywhs[x_box, y_box:y_box + 1],ctx=ctx)  # 两者是否重合
##        print(iou)
##        ignore_mask[x_box, y_box] = (iou < ignore_thresh).astype("float32").reshape(-1)                   # 把不重合的输出
##    return ignore_mask

if __name__=='__main__': 
    images_path = '../Data/demo'
    image_list = []
    classes_path  = '../Data/demo/train/coco.names'
    mode = 'train'
    label_mode = "txt"
    label_list = []
    classes = load_classes(classes_path)
    num_classes = len(classes)
    input_dim = 416
    pattern = re.compile("(.png|.jpg|.bmp|.jpeg)")

    # 载入标签的真实文件地址
    images_path = os.path.join(images_path, mode)
    images_path = os.path.join(images_path, "JPEGImages")
    # 先交换目录，从label得到标签回头找图像
    label_path = images_path.replace('JPEGImages', 'labels')
    label_list = os.listdir(label_path)
    label_list = [os.path.join(label_path, im.strip()) for im in label_list]
    #print(label_list)
    pattern = re.compile("(.txt)")
    
    # 载入图片的真实地址    
    for i in range(len(label_list)):
        image_name = pattern.sub(lambda s: ".jpeg", label_list[i]).replace('labels','JPEGImages')
        image_list.append(image_name)
        
    for idx in range(len(label_list)):
##    for idx in range(1):
        print(u'打开图片的url = {}'.format(image_list[idx]))
        print(u'标签的url = {}'.format(label_list[idx]))
        image = cv2.imread(image_list[idx])
        print(u'cv2提取图像原始横宽深：{}'.format(image.shape))
        label = prep_label(label_list[idx], classes=classes)  # debuging 2018-11-27
        label_org = label.copy()
        # 对百分比的x,y,w,h处理成具体数值
        (img_h, img_w,_) = image.shape
        label[:,0] =  label[:,0] * img_w
        label[:,1] =  label[:,1] * img_h
        label[:,2] =  label[:,2] * img_w
        label[:,3] =  label[:,3] * img_h
        show_jpg_result(url = image, key_array = label[:,[4,0,1,2,3]].asnumpy())     # 展示标签位置
        image, label = prep_image(image, input_dim, label_org)
        label_org = label.copy()
        print(u'变形后图像维度:{}'.format(image.shape))
        #print(u'变形后标签数值:{}'.format(label))
        # 对百分比的x,y,w,h处理成具体数值
        (img_h, img_w) = (416,416)
        label[:,0] =  label[:,0] * img_w
        label[:,1] =  label[:,1] * img_h
        label[:,2] =  label[:,2] * img_w
        label[:,3] =  label[:,3] * img_h
        #print(label)
        show_jpg_result(url = np.transpose(image,(1,2,0)), key_array = label[:,[4,0,1,2,3]].asnumpy())  # 展示标签位置
        label, true_xywhc = prep_final_label(label_org, len(classes), input_dim=input_dim)
        label_org = label.copy()
        print(u'经过标签最终标签预处理后t_y维度是:{}'.format(label.shape))
        print(u'经过标签最终标签预处理后含c=1的t_y数量 = {}'.format(nd.sum(label[:,4]==1)))
        print(u'经过标签最终标签预处理后true_xywhc.shape = {}'.format(true_xywhc.shape))
        print(u'经过标签最终标签预处理后含c=1的t_xywhs数量 = {}'.format(nd.sum(true_xywhc[:,4]==1)))
        t_xywhc_np = true_xywhc.asnumpy()
        print(u'经过标签最终标签预处理后含c=1的t_xywhs索引号 = {}'.format(np.where(t_xywhc_np[:,4]==1)))
        index = np.where(t_xywhc_np[:,4]==1)
        t_xywhc_np = t_xywhc_np[index[0],:]
        t_xywhc_np = t_xywhc_np[:,[4,0,1,2,3]]
        print(t_xywhc_np)
        print(label_org[index[0],:])
        show_jpg_result(url = np.transpose(image,(1,2,0)), key_array = t_xywhc_np)

##        ignore_mask = calculate_ignore(nd.expand_dims(label_org,axis=0), nd.expand_dims(true_xywhc,axis=0))
##        print('ignore_mask.shape = {}'.format(ignore_mask.squeeze().shape))
##        print('ignore_mask = {}'.format(ignore_mask.squeeze().size))
##        print(u'不用屏蔽的序号：{}'.format(np.where(ignore_mask.squeeze().asnumpy()==0)))

    
