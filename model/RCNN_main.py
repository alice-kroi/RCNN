import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms
import numpy as np

class RCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(RCNN, self).__init__()
        # 特征提取网络（使用预训练的VGG16）
        self.backbone = models.vgg16(pretrained=True).features
        self.fc_features = 512 * 7 * 7  # VGG16最后特征图尺寸
        
        # 分类器（对象类别 + 背景）
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes + 1)  # 包含背景类
        )
        
        # 边界框回归器
        self.bbox_regressor = nn.Sequential(
            nn.Linear(self.fc_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4)  # 输出坐标偏移量 (dx, dy, dw, dh)
        )

    def forward(self, regions, img_size):
        """
        参数：
            regions: 区域提议列表 [N, 4] (x1, y1, x2, y2)
            img_size: 原始图像尺寸 (H, W)
        返回：
            scores: 分类得分 [N, num_classes+1]
            bbox_deltas: 边界框偏移量 [N, 4]
        """
        # 特征提取
        features = self.backbone(regions)
        features = features.view(features.size(0), -1)
        
        # 分类和回归
        scores = self.classifier(features)
        bbox_deltas = self.bbox_regressor(features)
        
        return scores, bbox_deltas

# 辅助函数：区域建议（需要安装selectivesearch）
# 需要先安装：pip install selectivesearch
def get_region_proposals(image):
    import selectivesearch
    _, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=100)
    return np.array([list(region['rect']) for region in regions if region['size'] > 200])

if __name__ == "__main__":
    # 测试示例
    model = RCNN(num_classes=20)
    input_regions = torch.randn(32, 4)  # 假设有32个区域提议
    scores, bbox_deltas = model(input_regions, (256, 256))
    print("分类得分形状:", scores.shape)
    print("边界框偏移形状:", bbox_deltas.shape)