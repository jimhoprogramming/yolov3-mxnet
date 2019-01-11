# -*- coding: utf-8 -*-
import argparse
import os
import sys
import re
import time
sys.path.append('../yolov3-mxnet-master')
from random import shuffle
from mxnet import autograd
from darknet import DarkNet
from utils_for_demo import *
sys.path.append('../dress')
from load_data import show_jpg_result

def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images_path', type=str)
##    parser.add_argument("--train", dest='train_data_path',default="../Data/warm_up_train_20180222/train/img_url.txt", type=str)   # 应该 --train d:\demo
    parser.add_argument("--train", dest='train_data_path', default="../Data/demo/", type=str)
    parser.add_argument("--val", dest='val_data_path', type=str)
    #parser.add_argument("--coco_train", dest="coco_train", type=str)
    #parser.add_argument("--coco_val", dest="coco_val", type=str)
    parser.add_argument("--lr", dest="lr", help="learning rate", default=1e-4, type=float)
##    parser.add_argument("--classes", dest="classes",default="../Data/warm_up_train_20180222/train/coco.names", type=str)
    parser.add_argument("--classes", dest="classes", default="../Data/demo/train/coco.names", type=str)
    parser.add_argument("--prefix", dest="prefix", default="demo_random")
    parser.add_argument("--gpu", dest="gpu", help="gpu id", default=0, type=str)
    parser.add_argument("--dst_dir", dest='dst_dir', default="../Data/demo/models", type=str)
    parser.add_argument("--epoch", dest="epoch", default=20, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=4, type=int)
    parser.add_argument("--ignore_thresh", dest="ignore_thresh", default=0.5)
    parser.add_argument("--params", dest='params', help="mxnet params file",default="../Data/demo/models/demo_random_yolov3_mxnet.params", type=str)
##    parser.add_argument("--params", dest='params', help="mxnet params file",default="./data/yolov3.weights", type=str)
##    parser.add_argument("--params", dest='params', help="mxnet params file",default="ramdon", type=str)
    parser.add_argument("--input_dim", dest='input_dim', default=416, type=int)

    return parser.parse_args()


def calculate_ignore(prediction, true_xywhs, ignore_thresh):
    '''
        功能：计算被忽略的边界框是哪些
        输入：
            prediction:被预测的结果[8，10647,4+1+cls_num]=[8,10647,7]
            true_xywhs:实际标签的转换值[8，10647,4+1+cls_num]=[8,10647,5]
            ignore_thresh:百分比阀值0.5
        通过全局变量透传的参数：
                        输入维度416
                        9个表框尺寸列表
                        预测是否含有对象的结果张量[8=batch_num,10647,1]
                        
        输出:
            ignore_mask:[batch_num,10647,1]实际标签含有物品且预测位置不重合的边框中心所有锚框=1
    '''
    ctx = prediction.context
    tmp_pred = predict_transform(prediction, input_dim, anchors)  # 将预测结果转换成需要的标签形式，关键！[8,10647,7]

    # 生成一个带批次有否物品的1值张量[8,10647,1]
    ignore_mask = nd.ones(shape=pred_score.shape, dtype="float32", ctx = ctx)
    item_index = np.argwhere(true_xywhs.asnumpy()[:, :, 4] == 1.0) # 实际标签含物品才有效=索引号的列表
    
    for x_box, y_box in item_index:
        # 显示含有pc标志的锚框位置
        #print(tmp_pred[x_box, y_box:y_box + 1, :5])
        #print(true_xywhs[x_box, y_box:y_box + 1])
        iou = bbox_iou(tmp_pred[x_box, y_box:y_box + 1, :4], true_xywhs[x_box, y_box:y_box + 1],ctx=ctx)  # 两者是否重合
        #print('iou:{}'.format(iou))
        ignore_mask[x_box, y_box] = (iou < ignore_thresh).astype("float32").reshape(-1)                   # 把不重合的输出
    return ignore_mask


class YoloDataSet(gluon.data.Dataset):
    def __init__(self, images_path, classes, input_dim=416, is_shuffle=False, mode="train", coco_path=None):
        super(YoloDataSet, self).__init__()
        self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)]
        self.classes = classes
        self.input_dim = input_dim
        #self.label_mode = "xml"
        self.label_mode = 'txt'
        self.label_list = []                 # 放标签url的变量
        self.image_list = []


        # 得到目标标签文件的url列表
        
        if os.path.isdir(images_path):
            images_path = os.path.join(images_path, mode)
            images_path = os.path.join(images_path, "JPEGImages")
            # 先交换目录，从label得到标签回头找图像
            self.label_path = images_path.replace('JPEGImages', 'labels')
            self.label_list = os.listdir(self.label_path)
            self.label_list = [os.path.join(self.label_path, im.strip()) for im in self.label_list]  # 放图片名url的变量,以类别分好的目录名应该合并在同一目录JPEGImages下
        if is_shuffle:
            shuffle(self.label_list)
        #pattern = re.compile("(.png|.jpg|.bmp|.jpeg)")
        pattern = re.compile("(.txt)")
        print('point to image_data_dir root is {}'.format(images_path))    # debug 

        # 整理JPEGImage目录内部全部图像名称统一为jpeg
        self.__change_file_ext_name(images_path)
        
        # 得到目标图像的url列表
        for i in range(len(self.label_list)):
            if pattern.search(self.label_list[i]) is None or not os.path.exists(self.label_list[i]):
                self.label_list.pop(i)
                continue
            if self.label_mode == "txt":
                image_name = pattern.sub(lambda s: ".jpeg", self.label_list[i]).replace('labels','JPEGImages')

            if not os.path.exists(image_name):
                self.label_list.pop(i)
                continue
            self.image_list.append(image_name)

        print(u'打开图片的url = {}'.format(self.image_list[0]))
        print(u'标签的url = {}'.format(self.label_list[0]))
        
    def __change_file_ext_name(self, path):
        i = 1
        for f in os.listdir(path):
            a, b = os.path.splitext(f)
            if b == '.jpg' or b == '.jpeg' or b == '.JPEG':
                os.rename(os.path.join(path, f), os.path.join(path, a + os.extsep + 'jpeg'))
            i += 1
            print(i)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        '''
            返回三个张量:
                        图像[row,col,3];
                        三个尺度的零标签[10647=([13x13+26x26+52x52],(4+1+classes_num)),(4+1+classes_num)];
                        转化后标签[同上]
        '''
        image = cv2.imread(self.image_list[idx])
        label = prep_label(self.label_list[idx], classes=self.classes)  # debuging 2018-11-27
        #print('prep_label func label shape = {}'.format(label.shape))
##        # 对百分比的x,y,w,h处理成具体数值
##        (img_h, img_w,_) = image.shape
##        label[:,0] =  label[:,0] * img_w
##        label[:,1] =  label[:,1] * img_h
##        label[:,2] =  label[:,2] * img_w
##        label[:,3] =  label[:,3] * img_h
        image, label = prep_image(image, self.input_dim, label)
        #print('prep_image func label shape = {}'.format(label.shape))
        label, true_xywhc = prep_final_label(label, len(self.classes), input_dim=self.input_dim)
        #print('prep_final_label func label = {}'.format(label.shape))
        return nd.array(image).squeeze(), label.squeeze(), true_xywhc.squeeze()


if __name__ == '__main__':
    args = arg_parse()
    if args.images_path:
        args.train_data_path = args.images_path
        args.val_data_path = args.images_path
    classes = load_classes(args.classes)
    num_classes = len(classes)
    print('num_classes is %s'%(num_classes))    # debug
    ctx = mx.cpu()
    print('ctx context is %s'%(ctx))
    input_dim = args.input_dim
    batch_size = args.batch_size
    train_dataset = YoloDataSet(args.train_data_path, classes=classes, is_shuffle=True, mode="train", coco_path=None)
    train_dataloader = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloaders = {
        "train": train_dataloader
    }
    if args.val_data_path:
        val_dataset = YoloDataSet(args.val_data_path, classes=classes, is_shuffle=True, mode="val", coco_path=None)
        val_dataloader = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        dataloaders["val"] = val_dataloader

    obj_loss = LossRecorder('objectness_loss')
    cls_loss = LossRecorder('classification_loss')
    box_loss = LossRecorder('box_refine_loss')
    positive_weight = 1.0
    negative_weight = 1.0

    l2_loss = L2Loss(weight=2.)

    net = DarkNet(num_classes=num_classes, input_dim=input_dim)
    net.initialize(init=mx.init.Xavier(), ctx=ctx)                                          # 初始化神经网络
    if args.params.endswith(".params"):
        net.load_parameters(args.params)
        X = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx = ctx)
        net(X)
        print("weight.params url = {} loading ......".format(args.params))
    elif args.params.endswith(".weights"): 
        X = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx = ctx)                     # feed shape tip here
        net(X)      # debuging what is it ?
        print("weight.weights url = {} loading ......".format(args.params))
        net.load_weights(args.params, fine_tune=num_classes != 80)
    else:
        X = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx = ctx)
        net(X)
        print(u'随机初始化')
        #print("params {} load error!".format(args.params))
        #exit()

    net.hybridize()
    #mx.viz.print_summary(net)         # 显示网络结构
    #mx.viz.plot_network(net).view()        # 导出网络符号图 
    # for _, w in net.collect_params().items():
    #     if w.name.find("58") == -1 and w.name.find("66") == -1 and w.name.find("74") == -1:
    #         w.grad_req = "null"
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)])

    total_steps = int(np.ceil(len(train_dataset) / batch_size) - 1)
    print('trainingset len is %s and total_step is %s'%(len(train_dataset),total_steps))       # debug
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=[200 * total_steps], factor=0.1)
    optimizer = mx.optimizer.Adam(learning_rate=args.lr, lr_scheduler=schedule)
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)                      # 定义模型

    best_loss = 100000.
    early_stop = 0

    for epoch in range(args.epoch):
        if early_stop >= 5:
            print("train stop, epoch: {0}  best loss: {1:.3f}".format(epoch - 5, best_loss))
            break
        print('Epoch {} / {}'.format(epoch, args.epoch - 1))
        print('-' * 20)

        for mode in ["train", "val"]:
            tic = time.time()
            if mode == "val":
                if not args.val_data_path:
                    continue
                total_steps = int(np.ceil(len(val_dataset) / batch_size) - 1)
            else:
                total_steps = int(np.ceil(len(train_dataset) / batch_size) - 1)
            cls_loss.reset()
            obj_loss.reset()
            box_loss.reset()
            # 设置使用多个gpu
            print(u'每次dataloader行为的图片数量%s'%(len(dataloaders[mode])))
            print(dataloaders[mode])
            for i, batch in enumerate(dataloaders[mode]):
                cpu_Xs = unsplit_and_load(batch[0],ctx)
                cpu_Ys = unsplit_and_load(batch[1],ctx)
                cpu_Zs = unsplit_and_load(batch[2],ctx)
                
                with autograd.record(mode == "train"):
                    loss_list = []
                    batch_num = 0
                    for cpu_x, cpu_y, cpu_z in zip(cpu_Xs, cpu_Ys, cpu_Zs):
                        '''
                        print(['cpu_x shape is :',cpu_x.shape])      # [8，416,416,3]
                        print(cpu_x)
                        print(['cpu_y shape is :',cpu_y.shape])      # [8，13x13+26x26+52x52,4+1+cls_num]
                        print(cpu_y)
                        print(['cpu_z shape is :',cpu_z.shape])      # [8，13x13+26x26+52x52,4+1+cls_num]=转换后值
                        print(cpu_z)
                        '''
                        mini_batch_size = cpu_x.shape[0]
                        prediction = net(cpu_x)                     # 喂X=图片数据给模型。得到输出结果
                        #prediction = cpu_y.copy()         
                        print(u'预测后结果集的维度 = {}'.format(prediction.shape))   # 出错，检查net定义部分！  2018-11-14 已解决
                        pred_xywh = prediction[:, :, :4]
                        pred_score = prediction[:, :, 4:5]
                        pred_cls = prediction[:, :, 5:]
                        with autograd.pause():
                            # 生成不重合的预测框中心点列表
                            ignore_mask = calculate_ignore(prediction.copy(), cpu_z, args.ignore_thresh)
                            #ignore_mask = calculate_ignore(cpu_y, cpu_z, args.ignore_thresh)
                            true_box = cpu_y[:, :, :4]
                            true_score = cpu_y[:, :, 4:5]
                            true_cls = cpu_y[:, :, 5:]
                            coordinate_weight = true_score.copy()
                            score_weight = nd.where(coordinate_weight == 1.0,
                                                    nd.ones_like(coordinate_weight) * positive_weight,
                                                    nd.ones_like(coordinate_weight) * negative_weight)
                            box_loss_scale = 2. - cpu_z[:, :, 2:3] * cpu_z[:, :, 3:4] / float(args.input_dim ** 2)

                            # 显示含有pc标志的锚框位置
                            
                            print('pred_score:{}'.format(np.where(pred_score.asnumpy() > 0.8)))
                            '''
                            print('true_score:{}'.format(np.where(true_score.asnumpy()==1)))
                            tmp_pred = predict_transform(prediction.copy(), input_dim, anchors)
                            item_index = np.argwhere(prediction.asnumpy()[:, :, 4] == 1.0)
                            print(len(item_index))
                            if len(item_index) > 0 and len(item_index) < 3*batch_size :
                                for i in range(tmp_pred.shape[0]):
                                    key_array = []
                                    for index in item_index:
                                        if index[0] == i:
                                            key_array.append(tmp_pred.asnumpy()[i,index[1],:])
                                    show_jpg_result(url = np.transpose(cpu_x.asnumpy()[i],(1,2,0)), key_array = np.asarray(key_array)[:,[4,0,1,2,3]])
                            '''
                        # 计算损失函数
                        loss_xywh = l2_loss(pred_xywh, true_box, ignore_mask * coordinate_weight * box_loss_scale)  #一旦进入就一起 iou 内部print(box1)cpu出错                        
                        loss_conf = l2_loss(pred_score, true_score)                                                 #一旦进入就一起 iou 内部print(box1)cpu出错
                        loss_cls = l2_loss(pred_cls, true_cls, coordinate_weight)
                        t_loss_xywh = nd.sum(loss_xywh) / mini_batch_size
                        t_loss_conf = nd.sum(loss_conf) / mini_batch_size
                        t_loss_cls = nd.sum(loss_cls) / mini_batch_size

                        loss = t_loss_xywh + t_loss_conf + t_loss_cls
                        batch_num += len(loss)

                        # 显示损失汇总值
                        print('t_loss_xywh:{}'.format(t_loss_xywh.asnumpy()))
                        print('t_loss_conf:{}'.format(t_loss_conf.asnumpy()))
                        print('t_loss_cls:{}'.format(t_loss_cls.asnumpy()))
                        print('loss:{}'.format(loss.asnumpy()))

##                        if t_loss_conf.asnumpy() > 10 and t_loss_conf.asnumpy() < 15:
##                            exit()
                        if mode == "train":
                            loss.backward()                         # 计算梯度 
                        with autograd.pause():
                            loss_list.append(loss[0].asscalar())
                            cls_loss.update([t_loss_cls])
                            obj_loss.update([t_loss_conf])
                            box_loss.update([t_loss_xywh])

                trainer.step(batch_num, ignore_stale_grad=True)      # 启动训练
                
                if (i + 1) % int(total_steps / 5) == 0:
                    mean_loss = 0.
                    for l in loss_list:
                        mean_loss += l
                    mean_loss /= len(loss_list)
                    print("{0}  epoch: {1}  batch: {2} / {3}  loss: {4:.3f}"
                          .format(mode, epoch, i, total_steps, mean_loss))
                if (i + 1) % int(total_steps / 2) == 0:
                    total_num = nd.sum(coordinate_weight)
                    item_index = np.nonzero(true_score.asnumpy())
                    print("predict case / right case: {}".format((nd.sum(pred_score > 0.5) / total_num).asscalar()))
                    print((nd.sum(nd.abs(pred_score * coordinate_weight - true_score)) / total_num).asscalar())
            #nd.waitall()
            print('epoch, mode, cls_loss, obj_loss, box_loss, time.time sec')
            print(epoch, mode, cls_loss.get(), obj_loss.get(), box_loss.get(), time.time() - tic)
        loss = cls_loss.get()[1] + obj_loss.get()[1] + box_loss.get()[1]

        if loss < best_loss:
            early_stop = 0
            best_loss = loss
            #net.save_params("./models/{0}_yolov3_mxnet.params".format(args.prefix))
            net.save_parameters("{0}/{1}_yolov3_mxnet.params".format(args.dst_dir, args.prefix))
        else:
            early_stop += 1
