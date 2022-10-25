import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from tinySSD import TinySSD
from data_load import load_data
from utils.box import *
from utils.anchor import *

net = TinySSD(num_classes=1)
net = net.to('cuda')
# 加载模型参数
net.load_state_dict(torch.load('net_40.pkl'))
def display(img, output, threshold,name):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.savefig(name.split('/')[-1])

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to('cuda'))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]
            
files = glob.glob('detection/test/*.jpg')
for name in files:
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X)    
    display(img, output.cpu(), threshold=0.9,name=name)

'''def display(img, output, threshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.savefig('test.jpg')
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to('cuda'))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]
            
files = glob.glob('detection/test/*.jpg')
for name in files:
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X)    
    display(img, output.cpu(), threshold=0.1)
    break
    '''