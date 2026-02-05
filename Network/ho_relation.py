from matplotlib.pyplot import polar
import torch
import torch.nn as nn
from zmq import device
# from Network.model import resnet50
from torchvision.models import resnet50
import torchvision
import numpy as np
import timm
from Network.relation import HumanObjectRelationModule


class Custom_Model(torch.nn.Module):
    def __init__(self,pretrained,device):
        """
        
        """
        super(Custom_Model, self).__init__()

        self.resnet = timm.create_model('resnet50d', pretrained=True)
        # self.resnet = resnet50(pretrained=True)

        
        self.resnet1 = torch.nn.Sequential(*(list(self.resnet.children())[0:7]))
        self.resnet2 = torch.nn.Sequential(*(list(self.resnet.children())[7:8]))
        self.global_avg_pool = torch.nn.AvgPool2d(7,ceil_mode=True)
        # self.model2 = torch.nn.Sequential()
        self.lstm = torch.nn.LSTM(1024,512,num_layers=1, batch_first=True)
        self.device = device

        # self.global_avg_pool = torch.nn.AvgPool2d(kernel_size=(7,7),stride=2)
        self.fc1 = torch.nn.Linear(in_features= 2048, out_features=1024, bias=True)
        self.fc2 = torch.nn.Linear(in_features= 2048, out_features=1024, bias=True)
        self.relation = HumanObjectRelationModule()
        self.class_predictor = torch.nn.Linear(in_features= 1024, out_features=11, bias=True)
        self.ctx_class_predictor = torch.nn.Linear(in_features= 1024, out_features=11, bias=True)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.normal_( self.class_predictor.weight, mean=0, std=0.01)
        nn.init.normal_(self.ctx_class_predictor.weight, mean=0, std=0.01)

        
    def forward(self, x, gt_box = None, obj_box = None):
        """

        """
        
        # pdb.set_trace()
        gt_box = gt_box.reshape(gt_box.shape[1],gt_box.shape[2])
        
        obj_box = obj_box.reshape(obj_box.shape[1],obj_box.shape[2])
        all_rois = torch.row_stack((gt_box,obj_box))
        all_rois = all_rois.reshape(1,all_rois.shape[0],all_rois.shape[1])

        with torch.no_grad():
            feat = self.resnet1(x)
            
        pooled_feat = torchvision.ops.roi_align(feat, list(gt_box.unsqueeze(0)), output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)
        pooled_ctx_feat = torchvision.ops.roi_align(feat, list(obj_box.unsqueeze(0)), output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)

        top_feat =  self.resnet2(pooled_feat)
        top_ctx_feat =  self.resnet2(pooled_ctx_feat)

        top_feat = self.global_avg_pool(top_feat)
        top_ctx_feat = self.global_avg_pool(top_ctx_feat)

        # pdb.set_trace()
        top_feat = top_feat.flatten(1,3)
        top_feat = self.fc1(top_feat)
        
        top_ctx_feat = top_ctx_feat.flatten(1,3)
        top_ctx_feat = self.fc2(top_ctx_feat)


        relation_feat, relation_ctx_feat = self.relation(top_feat, top_ctx_feat, gt_box, obj_box)

        top_feat = top_feat + relation_feat
        top_ctx_feat = top_ctx_feat + relation_ctx_feat

        cls_pred = self.class_predictor(top_feat)
        ctx_cls_pred = self.ctx_class_predictor(top_ctx_feat)
        # pdb.set_trace()
        cls_pred = cls_pred.unsqueeze(0)
        ctx_cls_pred = ctx_cls_pred.unsqueeze(0)

        ctx_cls_pred = torch.max(ctx_cls_pred,1,keepdims=True)
        cls_pred = torch.add(cls_pred, ctx_cls_pred[0])
        
        return cls_pred

