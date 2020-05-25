from __future__ import absolute_import

import torch
from torch import nn
from  torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch import nn

__all__ = ['ResNet50','conv']

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)

def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_out')
        nn.init.constant_(m.bias,0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight,1.0)
            nn.init.constant_(m.bias,0.0)

class ResNet50(nn.Module):

    def __init__(self,num_classes,last_stride,model_path,neck,model_name,pretrain_choice,neck_feat=1):
        super(ResNet50,self).__init__()

        self.feat_dim = 2048
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride = (1,1)
        resnet50.layer4[0].downsample[0].stride = (1,1)
        backbone = nn.Sequential(*list(resnet50.children())[:-2])

        self.base = nn.Sequential(*backbone)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if self.neck == 'no':
            self.classifier = nn.Linear(self.feat_dim,self.num_classes)
            self.classifier.apply(weight_init_classifier)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.feat_dim)
            self.classifier = nn.Linear(self.feat_dim,self.num_classes,bias=False)
            self.bottleneck.apply(weight_init_kaiming)
            self.classifier.apply(weight_init_classifier)

    def forward(self,x,pid=None,camid=None,img_path=None):
        b,t,c,w,h = x.size()
        # print(x.size())
        x = x.view(b*t,c,w,h)
        feat = self.base(x)
        feat  = feat.view(b,t,self.feat_dim,feat.size(2),feat.size(3))
        feat = feat.permute(0,2,1,3,4)
        feat = self.pool(feat)
        feat = feat.view(b,self.feat_dim) #feat.size() = (b,feat_dim)
        # print(feat.size())

        if self.neck == 'no':
            feature = feat
        elif self.neck == 'bnneck':
            feature = self.bottleneck(feat)


        if self.training:
            cls_score = self.classifier(feature)
            return cls_score,feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return feat

class conv(nn.Module):

    def __init__(self,num_classes,last_stride,model_path,neck,model_name,pretrain_choice,neck_feat=1):
        super(ResNet50, self).__init__()

        self.feat_dim = 2048
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat







if __name__ == '__main__':
    x = torch.rand(16,5,3,256,128)
    num_classes = 625
    last_stride = 1
    model_path = '/home/wyq/.torch/models/resnet50-19c8e357.pth'
    neck = 'no'
    model_name = 'resnet50'
    pretrain_choice = 'imagenet'
    model = ResNet50(num_classes,last_stride,model_path,neck,model_name,pretrain_choice)
    model(x)