import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms

class RCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, num_classes=20):
        """
        目标检测分类器模块
        参数：
            input_dim: 输入特征维度 (VGG16特征提取后的维度512*7*7)
            hidden_dim: 全连接层隐藏维度 (默认4096)
            num_classes: 检测类别数 (包含背景为num_classes+1)
        """
        super(RCNNClassifier, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes + 1)  # 增加背景类别
        )
        
        self._init_weights()

    def _init_weights(self):
        """ Xavier初始化 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        输入：
            x: 特征张量 [batch_size, input_dim]
        输出：
            scores: 分类得分 [batch_size, num_classes+1]
        """
        return self.fc(x)

# 示例用法
if __name__ == "__main__":
    # 输入维度与VGG16特征提取后的维度一致 (512*7*7=25088)
    classifier = RCNNClassifier(input_dim=25088)
    dummy_input = torch.randn(32, 25088)  # 批次大小32
    output = classifier(dummy_input)
    print(f"输出尺寸: {output.shape}")  # 应得到 torch.Size([32, 21])