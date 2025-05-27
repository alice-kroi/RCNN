import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, anchor_num=9):
        super().__init__()
        self.anchor_num = anchor_num
        
        # 基础卷积层
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        # 分类分支（目标/非目标）
        self.cls_layer = nn.Conv2d(mid_channels, anchor_num * 2, kernel_size=1)
        
        # 回归分支（边界框修正）
        self.reg_layer = nn.Conv2d(mid_channels, anchor_num * 4, kernel_size=1)
        
        # 初始化参数
        for layer in [self.conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        # 共享卷积
        x = F.relu(self.conv(features))
        
        # 分类分数 [batch, anchor*2, H, W]
        cls_scores = self.cls_layer(x)
        
        # 回归参数 [batch, anchor*4, H, W]
        reg_params = self.reg_layer(x)
        
        return cls_scores, reg_params

    # 锚点生成方法（需要根据实际特征图尺寸实现）
    def generate_anchors(self, feature_shape, image_shape):
        """生成锚点框的基类方法"""
        pass  # 实际实现需要计算锚点坐标