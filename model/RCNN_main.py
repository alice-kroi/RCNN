import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms,RoIPool
import numpy as np
import selectivesearch
import cv2
from module.models.classifer import RCNNClassifier
from module.models.regressor import RCNNRegressor
from module.models.vgg16 import VGG
#from module.models.roi_pooling import RoiPooling



class RCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(RCNN, self).__init__()
        # 特征提取网络（使用预训练的VGG16）
        self.backbone = VGG(features=256)
        self.fc_features = 512 * 7 * 7  # VGG16最后特征图尺寸
        
        # 分类器（对象类别 + 背景）
        self.classifier = RCNNClassifier(self.fc_features, num_classes=num_classes)
        
        # 边界框回归器
        self.bbox_regressor = RCNNRegressor(self.fc_features)

        # ROI池化层（固定大小）
        self.roi_pool = RoIPool(
            output_size=(7, 7),
            spatial_scale=1.0/16  # 根据VGG的降采样倍数设置
        )

    def forward(self, images, rois):
        # 特征提取
        base_features = self.backbone(images)  # 假设返回形状 [B, 512, H', W']
        
        # ROI池化（处理坐标缩放）
        pooled_features = self.roi_pool(base_features, rois)
        
        # 后续处理保持不变
        flattened = pooled_features.view(pooled_features.size(0), -1)
        cls_scores = self.classifier(flattened)
        bbox_preds = self.bbox_regressor(flattened)
        
        return cls_scores, bbox_preds

# 辅助函数：区域建议（需要安装selectivesearch）
# 需要先安装：pip install selectivesearch
def get_region_proposals(image):
    import selectivesearch
    _, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=100)
    return np.array([list(region['rect']) for region in regions if region['size'] > 200])

if __name__ == "__main__":
    # 测试示例
    model = RCNN(num_classes=20)
    
    # 模拟输入数据
    batch_size = 2
    img_height = 256
    img_width = 256
    
    # 1. 生成模拟图像 [batch, channel, height, width]
    images = torch.randn(batch_size, 3, img_height, img_width)
    
    # 2. 生成区域提议 [N, 5] (batch_index, x1, y1, x2, y2)
    num_rois_per_image = 5
    rois = []
    for batch_idx in range(batch_size):
        # 生成随机坐标（确保x2 > x1，y2 > y1）
        x1 = torch.randint(0, 100, (num_rois_per_image,))
        y1 = torch.randint(0, 100, (num_rois_per_image,))
        x2 = x1 + torch.randint(50, 150, (num_rois_per_image,))
        y2 = y1 + torch.randint(50, 150, (num_rois_per_image,))
        
        # 添加batch索引并堆叠
        batch_indices = torch.full((num_rois_per_image, 1), batch_idx)
        rois_per_image = torch.cat([
            batch_indices.float(),
            x1.unsqueeze(1).float(),
            y1.unsqueeze(1).float(),
            x2.unsqueeze(1).float(),
            y2.unsqueeze(1).float()
        ], dim=1)
        rois.append(rois_per_image)
    
    rois = torch.cat(rois, dim=0)
    
    # 3. 前向传播
    cls_scores, bbox_deltas = model(images, rois)
    
    # 4. 验证输出形状
    print("\n测试结果：")
    print(f"分类得分形状: {cls_scores.shape} (应为 [{batch_size*num_rois_per_image}, 21])")
    print(f"边界框偏移形状: {bbox_deltas.shape} (应为 [{batch_size*num_rois_per_image}, 4])")
    
    # 5. 可选：显示部分结果
    print("\n前3个分类得分示例：")
    print(cls_scores[:3])
    print("\n前3个边界框偏移示例：")
    print(bbox_deltas[:3])