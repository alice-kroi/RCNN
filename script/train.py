import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms,RoIPool
from module.models.vgg16 import VGG
from module.models.classifer import RCNNClassifier
from module.models.regressor import RCNNRegressor
from tool.dataset_load import CelebaDetectionDataset,load_config
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR
from model.RCNN_main import RCNN




if __name__ == "__main__":
    # 加载配置
    config = load_config("config.yaml")
    
    # 初始化数据集和数据加载器
    train_dataset = CelebaDetectionDataset(
        root_dir=config['dataset_path'],
        transform=None  # 根据实际情况添加预处理
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # 初始化模型
    model =RCNN
    
    # 定义损失函数和优化器
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
    ], lr=0.001, momentum=0.9)

    # 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(100):
        for images, targets in train_loader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # 前向传播
            cls_scores, bbox_preds = model(images)
            
            # 计算损失
            # 这里需要根据实际标注格式计算损失
            # 以下为示例逻辑，需根据实际数据调整
            loss_cls = criterion_cls(cls_scores, targets['labels'])
            loss_reg = criterion_reg(bbox_preds, targets['boxes'])
            total_loss = loss_cls + loss_reg
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss.item()}")