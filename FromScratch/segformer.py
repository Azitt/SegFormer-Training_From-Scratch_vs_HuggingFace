from encoder import mix_transformer
from decoder import segformer_head
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
from timm.models.layers import drop_path, trunc_normal_

class segformer_mit_b3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.backbone = mix_transformer(in_chans=in_channels, embed_dims=(64, 128, 320, 512),
                                    num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                    sr_ratios=(8, 4, 2, 1), dropout_p=0.0, drop_path_p=0.1)
        self.decoder_head = segformer_head(in_channels=(64, 128, 320, 512),
                                    num_classes=num_classes, embed_dim=256)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x) #: Call Encoder
        x = self.decoder_head(x) #: Call Decoder
        x = F.interpolate(x, size=image_hw, mode='bilinear', align_corners=False) # Interpolate to output size
        return x
