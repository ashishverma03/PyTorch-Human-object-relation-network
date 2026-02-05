from matplotlib.pyplot import polar
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
# from Network.model import resnet50
from torchvision.models import resnet50
import torchvision
import numpy as np
import pdb

class HumanObjectRelationModule(torch.nn.Module):
    def __init__(self, num_feat=1024, num_group=16):
        """
        
        """
        super(HumanObjectRelationModule, self).__init__()
        self.num_feat = num_feat
        self.num_group = num_group
        self.dim_k = int(num_feat / num_group)

        self.fc_gt_ctx_position = torch.nn.Linear(in_features= 64, out_features=num_group, bias=True)
        self.fc_ctx_gt_position = torch.nn.Linear(in_features= 64, out_features=num_group, bias=True)
        self.fc_gt = torch.nn.Linear(in_features= 1024, out_features=num_feat, bias=True)
        self.fc_ctx = torch.nn.Linear(in_features= 1024, out_features=num_feat, bias=True)
        self.gt_ctx_linear_out = nn.Conv2d(16384, self.num_feat, 1, 1, 0, groups=self.num_group)
        self.ctx_gt_linear_out = nn.Conv2d(16384, self.num_feat, 1, 1, 0, groups=self.num_group)

        nn.init.normal_(self.fc_gt_ctx_position.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc_ctx_gt_position.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc_gt.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc_ctx.weight, mean=0, std=0.01)


        
    def forward(self, feat, ctx_feat, box, ctx_box):
        """

        """
        # pdb.set_trace()
        gt_ctx_pos_embedding = F.relu(self.position_embedding(box, ctx_box, feat_dim = 64))
        gt_ctx_pos_feat = self.fc_gt_ctx_position(gt_ctx_pos_embedding)  # (M*N, num_group)
        gt_ctx_pos_feat = gt_ctx_pos_feat.transpose(1,0)  # (num_group, M*N)

        ctx_gt_pos_embedding = F.relu(self.position_embedding(ctx_box, box, feat_dim=64))  # (N*M, feat_dim)
        ctx_gt_pos_feat = self.fc_ctx_gt_position(ctx_gt_pos_embedding)  # (N*M, num_group)
        ctx_gt_pos_feat = ctx_gt_pos_feat.transpose(1,0)  # (num_group, M*N)

        gt_data = self.fc_gt(feat)
        gt_data = gt_data.reshape((gt_data.shape[0], self.num_group, self.dim_k )).permute(1,0,2)
        ctx_data = self.fc_ctx(ctx_feat)
        ctx_data = ctx_data.reshape((ctx_data.shape[0], self.num_group, self.dim_k )).permute(1,0,2)

        gt_ctx = torch.matmul(gt_data,ctx_data.permute(0,2,1))
        gt_ctx = (1.0 / math.sqrt(float(self.dim_k))) * gt_ctx
        ctx_gt = gt_ctx.permute(0,2,1)

        gt_ctx_pos_feat = gt_ctx_pos_feat.view_as( gt_ctx)
        gt_ctx = gt_ctx.permute(1,0,2)
        gt_ctx_pos_feat = gt_ctx_pos_feat.permute(1, 0, 2)  # (M, num_group, N)

        weighted_gt_ctx = torch.log(torch.clamp(gt_ctx_pos_feat, 1e-6)) + gt_ctx
        weighted_gt_ctx = F.softmax(weighted_gt_ctx, dim=2)
        weighted_gt_ctx = torch.flatten(weighted_gt_ctx, start_dim=0, end_dim=1)

        gt_output = torch.matmul(weighted_gt_ctx, ctx_feat)
        gt_output = gt_output.reshape((-1,self.num_group*self.num_feat,1,1))
        gt_output = self.gt_ctx_linear_out(gt_output)  # (M, 1024, 1, 1)

        ctx_gt_pos_feat = ctx_gt_pos_feat.view_as( ctx_gt)
        ctx_gt = ctx_gt.permute(1,0,2)
        ctx_gt_pos_feat = ctx_gt_pos_feat.permute(1, 0, 2)  # (M, num_group, N)

        weighted_ctx_gt = torch.log(torch.clamp(ctx_gt_pos_feat, 1e-6)) + ctx_gt
        weighted_ctx_gt = F.softmax(weighted_ctx_gt, dim=2)
        weighted_ctx_gt = torch.flatten(weighted_ctx_gt, start_dim=0, end_dim=1)
        
        ctx_output = torch.matmul(weighted_ctx_gt, feat)
        ctx_output = ctx_output.reshape((-1,self.num_group*self.num_feat,1,1))
        ctx_output = self.gt_ctx_linear_out(ctx_output)  # (M, 1024, 1, 1)

        return  gt_output.flatten(1,3),  ctx_output.flatten(1,3)
        
    def position_embedding(self, box, ctx_box, feat_dim=64, wave_length=1000):

        """Compute position embedding.

        Parameters
        ----------
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.
        feat_dim: int, default is 64
        wave_length: int default is 1000

        Returns
        -------
        embedding
            Returns (M, N, feat_dim).
        """
        # position encoding
        # (M, 1)
        # pdb.set_trace()
        xmin, ymin, xmax, ymax = torch.chunk(box, chunks=4, dim=1)
        box_width = xmax - xmin + 1.
        box_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # (N, 1)
        ctx_xmin, ctx_ymin, ctx_xmax, ctx_ymax = torch.chunk(ctx_box, chunks=4, dim=1)
        ctx_box_width = ctx_xmax - ctx_xmin + 1.
        ctx_box_height = ctx_ymax - ctx_ymin + 1.
        ctx_center_x = 0.5 * (ctx_xmin + ctx_xmax)
        ctx_center_y = 0.5 * (ctx_ymin + ctx_ymax)

        # (M, N)
        delta_x = torch.sub(center_x, torch.transpose(ctx_center_x,1,0))
        delta_x = torch.div(delta_x, box_width)
        delta_x = torch.log(torch.clamp(torch.abs(delta_x), min=1e-3))
        delta_y = torch.sub(center_y, torch.transpose(ctx_center_y,1,0))
        delta_y = torch.div(delta_y, box_height)
        delta_y = torch.log(torch.clamp(torch.abs(delta_y), min=1e-3))
        delta_width = torch.div(torch.transpose(ctx_box_width,1,0), box_width)
        delta_width = torch.log(delta_width)
        delta_height = torch.div(torch.transpose(ctx_box_height,1,0), box_height)
        delta_height = torch.log(delta_height)
        # (M, N, 4)
        position_mat = torch.stack((delta_x, delta_y, delta_width, delta_height), 2)

        # # position embedding
        feat_range = torch.arange(0, feat_dim / 8).cuda()
        dim_mat = torch.pow(torch.tensor([wave_length]).cuda(), (8/ feat_dim) * feat_range)
        dim_mat = torch.reshape(dim_mat, (1,1,1,-1))  # (1, 1, 1, feat_dim/8)
        # position_mat (M, N, 4, 1)
        position_mat = (100.0 * position_mat).unsqueeze(-1)
        div_mat = torch.div(position_mat, dim_mat)  # (M, N, 4, feat_dim/8)
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        embedding = torch.cat((sin_mat, cos_mat), 3)   # (M, N, 4, feat_dim/4)
        return embedding.reshape((embedding.shape[0]*embedding.shape[1],64))


