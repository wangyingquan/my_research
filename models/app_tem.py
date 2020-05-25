import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def init_pretrained_weight(model,model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url, model_dir = './')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)


class app_tem(nn.Module):

    def __init__(self, num_classes,  last_stride, model_path, model_name, pretrain_choice, global_refine_method, local_refine_method):
        super(app_tem, self).__init__()
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base,model_urls[model_name])
            print('Loading pretrained ImageNet model......')

        self.global_refine_method = global_refine_method
        self.local_refine_method = local_refine_method
        self.num_classes = num_classes
        self.part_num = 4
        self.feat_pool = nn.AdaptiveAvgPool2d((1,1))
        self.local_part_avgpool = nn.AdaptiveAvgPool2d((self.part_num,1))

        #temporal block
        self.compact_conv_1 = nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.compact_conv_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.compact_conv_3 = nn.Conv2d(in_channels = 512, out_channels = 2048, kernel_size = 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(2048)

        self.temporal_lstm = nn.LSTM(input_size = 2048, hidden_size = 1024,num_layers = 1, bidirectional = True)
        self.temporal_conv = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 1, stride = 1, padding = 1)
        self.cat_fc = nn.Linear(in_features = 4096, out_features = 2048)

        self.appearance_bottleneck = nn.BatchNorm1d(2048)
        self.appearance_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.appearance_bottleneck.bias.requires_grad_(False)

        self.temporal_bottleneck = nn.BatchNorm1d(2048)
        self.temporal_classifier = nn.Linear(in_features=2048, out_features=self.num_classes, bias=False)
        self.temporal_bottleneck.bias.requires_grad_(False)

        self.sum_bottleneck = nn.BatchNorm1d(2048)
        self.sum_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.sum_bottleneck.bias.requires_grad_(False)

        self.temporal_conv.apply(weights_init_kaiming)
        self.compact_conv_1.apply(weights_init_kaiming)
        self.compact_conv_2.apply(weights_init_kaiming)
        self.compact_conv_3.apply(weights_init_kaiming)

        self.appearance_bottleneck.apply(weights_init_kaiming)
        self.temporal_classifier.apply(weight_init_classifier)

        self.temporal_bottleneck.apply(weights_init_kaiming)
        self.temporal_classifier.apply(weight_init_classifier)

        self.cat_fc.apply(weight_init_classifier)
        self.sum_bottleneck.apply(weights_init_kaiming)
        self.sum_classifier.apply(weight_init_classifier)

    def global_Center_cosine(self,feat_vec):
        b, t, _ = feat_vec.size()

        similar_matrix = torch.zeros(b, t, t)

        for i in range(t):
            for j in range(t):
                similar_matrix[:, i, j] = torch.cosine_similarity(feat_vec[:, i, :], feat_vec[:, j, :])

        similar_score = torch.sum(similar_matrix, 2, keepdim=True)
        remove_id = torch.argmin(similar_score, 1)
        refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))

        for i in range(b):
            refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]  # b*t-1*10

        #L2 normalize
        refine_feature = refine_feature.cuda()  #(32,3,2048)
        norm_score = torch.norm(refine_feature, 2, dim=2).unsqueeze(2)
        refine_feature = refine_feature / norm_score
        #L1 normalize
        #norm_score = torch.norm(refine_feature, 1, dim=2).unsqueeze(2)
        #refine_feature = refine_feature / norm_score

        #或者把度量的距离改成欧式距离
        # cosine_sum_similar = 0
        # for i in range(t - 1):
        #     for j in range(i + 1, t - 1):
        #         cosine_similar_score = torch.cosine_similarity(refine_feature[:, i, :], refine_feature[:, j, :])
                #归一化处理：cosine_similar_score = torch.div(cosine_similar_score + 1, 2)
                # cosine_similar_score = torch.sigmoid(cosine_similar_score)
                # cosine_similar_score = -torch.log(cosine_similar_score)
                # cosine_sum_similar = cosine_sum_similar + cosine_similar_score

        return refine_feature

    def global_KNN_cosine(self,feat_vec):  # 这个要要验证一下，cosine_similarity的输出是不是可以跨维度的。   r()eply:cosine_similarity是可以跨维度，或者说可以保持维度
        b, t, _ = feat_vec.size()
        refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))
        feat_vec_avg = torch.mean(feat_vec, 1)
        similar_matrix = torch.zeros(b, t)

        for i in range(t):
            similar_score = torch.cosine_similarity(feat_vec_avg, feat_vec[:, i, :])
            similar_matrix[:, i] = similar_score

        remove_id = torch.argmin(similar_matrix, 1)

        for i in range(b):
            refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]  # b*t-1*1024

        refine_feature = refine_feature.cuda()  # (32,3,2048)
        norm_score = torch.norm(refine_feature, 2, dim=2).unsqueeze(2)
        refine_feature = refine_feature / norm_score

        # cosine_sum_similar = 0
        # for i in range(t - 1):
        #     for j in range(i + 1, t - 1):
        #         cosine_similar_score = torch.cosine_similarity(refine_feature[:, i, :], refine_feature[:, j, :])
        #         cosine_similar_score = torch.sigmoid(cosine_similar_score)         # 成比例压缩到(0,1)区间
        #         cosine_similar_score = - torch.log(cosine_similar_score)
        #         cosine_sum_similar = cosine_sum_similar + cosine_similar_score

        return refine_feature

    def local_Center_cosine(self, feat_vec):
        b, t, c, n = feat_vec.size()
        similar_matrix = torch.zeros(b, n, t, t)

        for i in range(n):
            for j in range(t):
                for k in range(t):
                    similar_matrix[:, i, j, k] = torch.cosine_similarity(feat_vec[:, j, :, i], feat_vec[:, k, :, i])

        similar_matrix = torch.sum(similar_matrix, 2)
        remove_id = torch.argmax(similar_matrix, 2)
        refine_local_feature = torch.zeros(b, t - 1, c, n)

        for i in range(b):
            for j in range(n):
                refine_local_feature[i, :, :, j] = feat_vec[i, torch.arange(t) != remove_id[i,j], :, j]

        refine_local_feature = refine_local_feature.permute(0,1,3,2)
        refine_local_feature = torch.mean(refine_local_feature, 2).cuda()

        norm_score = torch.norm(refine_local_feature, 2, dim = 2).unsqueeze(2)
        refine_local_feature = refine_local_feature / norm_score

        return refine_local_feature

    def appearance(self,feat_map):
        b, t, c, w, h = feat_map.size()
        feat_map = feat_map.view(b*t, c, w, h)
        feat_local_vec = self.local_part_avgpool(feat_map)
        feat_local_vec = feat_local_vec.view(b, t, -1, self.part_num)

        if self.local_refine_method == 'Center_cosine':
            refine_local_feature = self.local_Center_cosine(feat_local_vec)
        elif self.local_refine_method == 'KNN_cosine':
            pass

        feat_global_vec = self.feat_pool(feat_map)
        feat_global_vec = feat_global_vec.view(b, t, -1)  #(b,t,1024)

        if self.global_refine_method == 'KNN_cosine':
            refine_global_feature = self.global_KNN_cosine(feat_global_vec)
        elif self.global_refine_method == 'Center_cosine':
            refine_global_feature = self.global_Center_cosine(feat_global_vec)

        refine_feature = 0.5 * refine_local_feature + 0.5 * refine_global_feature
        cosine_sum_similar = 0
        for i in range(t - 1):
            for j in range(i + 1, t - 1):
                cosine_similar_score = torch.cosine_similarity(refine_feature[:, i, :], refine_feature[:, j, :])
                # 归一化处理：cosine_similar_score = torch.div(cosine_similar_score + 1, 2)
                cosine_similar_score = torch.sigmoid(cosine_similar_score)
                cosine_similar_score = -torch.log(cosine_similar_score)
                cosine_sum_similar = cosine_sum_similar + cosine_similar_score

        credible_feature = torch.mean(refine_global_feature,1)
        appearance_loss = torch.mean(cosine_sum_similar).cuda()
        return credible_feature, appearance_loss

    def temporal_block(self, gap_feature_map):
        # short_cut = gap_feature_map

        gap_feature_map = self.compact_conv_1(gap_feature_map)
        gap_feature_map = self.bn1(gap_feature_map)

        gap_feature_map = self.compact_conv_2(gap_feature_map)
        gap_feature_map = self.bn2(gap_feature_map)

        gap_feature_map = self.compact_conv_3(gap_feature_map)
        gap_feature_map = torch.relu(self.bn3(gap_feature_map))

        return gap_feature_map

    def temporal(self, feat_map):
        b, t, c, h, w = feat_map.size()
        feat_map_clone = feat_map.detach()
        gap_feat_map = []

        for i in range(t - 1):
            gap_feat = feat_map[:, i, :, :, :] - feat_map_clone[:, i + 1, :, :, :]
            gap_feat_map.append(gap_feat)

        gap_feat_map = torch.stack(gap_feat_map, 1)
        gap_feat_map = gap_feat_map ** 2
        gap_feat_map = gap_feat_map.view(b * (t - 1), c, h, w)
        gap_feat_map = self.temporal_block(gap_feat_map)  #(96, 2048, 16,  8)

        gap_feat_map = gap_feat_map.view(b, t - 1, c, h, w)                      #(32, 3, 2048, 16, 8)
        gap_feat_map = gap_feat_map.permute(0, 2, 1, 3, 4)                       #(32, 2048, 3, 16, 8)
        gap_feat_map = self.temporal_conv(gap_feat_map.view(b * c, t-1, w, h))   #(32x2048, 3, 16, 8)

        gap_feat_vector = self.feat_pool(gap_feat_map)                           #(32x2048, 3, 1, 1)
        gap_feat_vector = gap_feat_vector.view(b, c, t-1)

        temporal_feat_vector , _ = self.temporal_lstm(gap_feat_vector)
        temporal_feat_vector = temporal_feat_vector.permute(1, 0, 2)
        temporal_feat_vector = torch.mean(temporal_feat_vector, 1)

        return temporal_feat_vector

    def forward(self, x, pids = None, camids = None):

        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)  # (b*t,c,16,8)
        feat_map = feat_map.view(b, t, self.in_planes, feat_map.size(2), feat_map.size(3))
        appearance_feature, appearance_loss = self.appearance(feat_map)
        temporal_feature = self.temporal(feat_map)

        BN_appearance_feature = self.appearance_bottleneck(appearance_feature)
        BN_temporal_feature = self.temporal_bottleneck(temporal_feature)

        feature = torch.cat([appearance_feature, temporal_feature], 1)
        feature = self.cat_fc(feature)
        BN_feature = self.sum_bottleneck(feature)

        cls_socre_list = []

        if self.training:
            cls_socre_list.append(self.appearance_classifier(BN_appearance_feature))
            cls_socre_list.append(self.temporal_classifier(BN_temporal_feature))
            cls_socre_list.append(self.sum_classifier(BN_feature))
            return appearance_loss, cls_socre_list, feature
        else:
            return BN_feature, pids.data.cpu(), camids.data.cpu()

if __name__ == '__main__':
    x = torch.rand(16,5,3,256,128)
    num_classes = 625
    last_stride = 1
    model_path = '/home/wyq/.torch/models/resnet50-19c8e357.pth'
    neck = 'no'
    model_name = 'resnet50'
    pretrain_choice = 'imagenet'
    model = app_tem(num_classes,last_stride,model_path,neck,model_name,pretrain_choice)
    model(x)