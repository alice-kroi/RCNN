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
    def __init__(self, num_classes=20,max_proposals=200):
        super(RCNN, self).__init__()
        # 特征提取网络（使用预训练的VGG16）
        self.backbone = VGG()
        self.fc_features = 512 * 7 * 7  # VGG16最后特征图尺寸
        
        # 分类器（对象类别 + 背景）
        self.classifier = RCNNClassifier(self.fc_features,hidden_dim=4096, num_classes=num_classes)
        
        # 边界框回归器
        self.bbox_regressor = RCNNRegressor(self.fc_features)

        # ROI池化层（固定大小）
        self.roi_pool = RoIPool(
            output_size=(7, 7),
            spatial_scale=1.0/16  # 根据VGG的降采样倍数设置
        )
        self.proposal_params = {
            'scale': max_proposals,
            'sigma': 0.9,
            'min_size': 200
        }
        self.max_proposals = max_proposals  # 新增最大建议数参数
    def forward(self, images):
        # 特征提取
        base_features = self.backbone(images)  # 假设返回形状 [B, 512, H', W']
        print("特征提取层后：",base_features.shape)
        # 生成区域建议（使用内置函数）

        batch_rois = []
        for batch_idx in range(images.size(0)):
            img_np = images[batch_idx].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            regions = self._generate_proposals(img_np)
            # 截取前N个建议区域
            regions = regions[:self.max_proposals]  # 新增截断逻辑
            rois = self._format_rois(regions, batch_idx)
            batch_rois.append(rois)
        
        rois_tensor = torch.cat(batch_rois).to(images.device)

        # ROI池化（处理坐标缩放）
        pooled_features = self.roi_pool(base_features, rois_tensor)
        print("roi池化层后：",pooled_features.shape)
        # 后续处理保持不变
        flattened = pooled_features.view(pooled_features.size(0), -1)
        print("展平后：",flattened.shape)

        cls_scores = self.classifier(flattened)
        bbox_preds = self.bbox_regressor(flattened)
        print("分类器后：",cls_scores.shape)
        print("回归器后：",bbox_preds.shape)
        return cls_scores, bbox_preds
    # 辅助方法
    def _generate_proposals(self, image):
        """内部区域建议生成"""
        _, regions = selectivesearch.selective_search(
            image, 
            scale=self.proposal_params['scale'],
            sigma=self.proposal_params['sigma'],
            min_size=self.proposal_params['min_size']
        )
        return [region['rect'] for region in regions if region['size'] > 200]

    def _format_rois(self, regions, batch_idx):
        """调整格式为[N, 4]"""
        rois = []
        for x, y, w, h in regions[:self.max_proposals]:  # 确保不超过最大提议数
            rois.append([x, y, x+w, y+h])
        
        # 填充零保持维度一致
        if len(rois) < self.max_proposals:
            padding = [[0,0,0,0]] * (self.max_proposals - len(rois))
            rois += padding
        
        return torch.tensor([[batch_idx] + coord for coord in rois], dtype=torch.float32)
__all__=['RCNN']
# 辅助函数：区域建议（需要安装selectivesearch）
# 需要先安装：pip install selectivesearch
def get_region_proposals(image):
    import selectivesearch
    _, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=100)
    return np.array([list(region['rect']) for region in regions if region['size'] > 200])

if __name__ == "__main__":

    # 测试示例
    model = RCNN(num_classes=20)
    
    # 1. 生成符合selective search要求的模拟图像（新增预处理）
    batch_size = 2
    img_height = 224
    img_width = 224
    
    # 使用0-255范围的图像数据（修改生成方式）
    images = (torch.rand(batch_size, 3, img_height, img_width) * 255).float()
    
    # 3. 前向传播（删除手动生成rois步骤）
    cls_scores, bbox_predict = model(images)
    
    # 4. 验证输出形状（更新验证逻辑）
    num_rois = cls_scores.shape[0]
    print("\n测试结果：")
    print(f"生成的ROI数量: {num_rois}")
    print(f"分类得分形状: {cls_scores.shape} (应为 [N, 21])")
    print(f"边界框偏移形状: {bbox_predict.shape} (应为 [N, 4])")
    
    # 5. 可选：显示部分结果（保持原样）
    print("\n前3个分类得分示例：")
    print(cls_scores[:3])
    print("\n前3个边界框偏移示例：")
    print(bbox_predict[:3])

    # 6. 保存模型（保持原样）
    torch.save(model.state_dict(), 'rcnn_model.pth')
    print("\n模型已保存为 rcnn_model.pth")