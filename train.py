import torch
import torch.nn as nn
from d2l import torch as d2l
from tinySSD import TinySSD
from data_load import load_data
from utils.box import *
from utils.anchor import *
batch_size = 32
train_iter = load_data(batch_size)
net = TinySSD(num_classes=1)
net = net.to('cuda')
#训练参数设置 主函数

def smooth_l1(x, sigma=1.0):
    sigma2 = sigma ** 2
    return torch.where(x.abs() < 1. / sigma2, 0.5 * x ** 2 * sigma2, x.abs() - 0.5 / sigma2)

def focal_loss(gamma, x):
    gamma=2
    return -(1 - x) ** gamma * torch.log(x)


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = focal_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = smooth_l1((bbox_preds - bbox_labels) * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

class Accumulator:
    """
    在‘n’个变量上累加
    """
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def _getitem_(self,idx):
        return self.data[idx]
    
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

num_epochs = 100
for epoch in range(num_epochs):
    print('epoch: ', epoch)
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = Accumulator(4)
    net.train()
    for features, target in train_iter:
        trainer.zero_grad()
        X, Y = features.to('cuda'), target.to('cuda')
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                    bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')


    # 保存模型参数
    if (epoch+1) % 10 == 0:
        torch.save(net.state_dict(), 'net_' + str(epoch+1) + '.pkl')