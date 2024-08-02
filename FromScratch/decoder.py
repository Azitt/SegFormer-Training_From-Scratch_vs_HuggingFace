import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
from timm.models.layers import drop_path, trunc_normal_

class segformer_head(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # 1x1 conv to fuse multi-scale output from encoder
        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1))
                                     for chans in reversed(in_channels)])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)

        # 1x1 conv to get num_class channel predictions
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]

        # Project each encoder stage output to the unified number of channels
        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]

        # Upsample each projected feature map to the common spatial resolution
        x = [F.interpolate(xi, size=feature_size, mode='bilinear', align_corners=False)
             for xi in x[:-1]] + [x[-1]]

        # Concatenate projected outputs
        concat_features = torch.cat(x, dim=1)

        # Apply linear fusion
        fused_features = self.linear_fuse(concat_features)
        fused_features = self.bn(fused_features)
        fused_features = F.relu(fused_features, inplace=True)
        fused_features = F.dropout(fused_features, p=self.dropout_p, training=self.training)

        # Project to the number of output classes
        output = self.linear_pred(fused_features)

        return output