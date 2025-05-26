import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms,RoIPool
from tool.dataset_load import CelebaDetectionDataset,load_config
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR
from model.RCNN_main import RCNN
import torchvision.transforms as transforms



if __name__ == "__main__":
    # 加载配置
    '''
    {'dataset': {'path': 'E:/github/dataset', 'image_size': 224, 'use_augmentation': True}, 'training': {'batch_size': 32, 'num_workers': 4, 'epochs': 100, 'device': 'cpu', 'save_interval': 10}, 'optimizer': {'type': 'sgd', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0005}, 'model': {'num_classes': 2, 'roi_output_size': 7, 'spatial_scale': 0.0625}, 'region_proposal': {'min_size': 200, 'scale': 500, 'sigma': 0.9}}    
    '''
    config = load_config("E:/github/RCNN/config/config.yaml")
    dataset_path = config['dataset']['path']
    image_size = config['dataset']['image_size']
    use_augmentation = config['dataset']['use_augmentation']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    epochs = config['training']['epochs']
    device = config['training']['device']
    save_interval = config['training']['save_interval']
    optimizer_type = config['optimizer']['type']
    lr = config['optimizer']['lr']
    momentum = config['optimizer']['momentum']
    weight_decay = config['optimizer']['weight_decay']
    num_classes = config['model']['num_classes']
    roi_output_size = config['model']['roi_output_size']
    spatial_scale = config['model']['spatial_scale']
    min_size = config['region_proposal']['min_size']
    scale = config['region_proposal']['scale']
    sigma = config['region_proposal']['sigma']


    print(config)

    # 初始化数据集和数据加载器
    train_dataset = CelebaDetectionDataset(
        root_dir=dataset_path,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip() if use_augmentation else None
        ])
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 使用配置中的batch_size
        shuffle=True,
        num_workers=num_workers  # 使用配置中的num_workers
    )

    for images, targets in train_loader:
        print(images.shape)  # 输出图像的形状
        print(targets)  # 输出标注信息
        break  # 只输出第一个批次的信息

    # 初始化模型
    model = RCNN(num_classes=num_classes)
    
    # 定义损失函数和优化器
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(100):
        for images, targets in train_loader:
            images = images.to(device)
            #targets = [t.to(device) for t in targets]
            
            # 前向传播
            cls_scores, bbox_preds = model(images)
            
            # 计算损失
            # 这里需要根据实际标注格式计算损失
            #loss_cls = criterion_cls(cls_scores, targets['labels'])
            print('原始标签形状',targets.shape)
            print('原始预测形状',bbox_preds.shape)

            loss_reg = criterion_reg(bbox_preds, targets)
            #total_loss = loss_cls + loss_reg
            total_loss = loss_reg
            # 反向传播
            print('loss:',total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        # 模型保存（使用提取的save_interval）
        if (epoch+1) % save_interval == 0:
            torch.save(model.state_dict(), f"rcnn_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1}, Loss: {total_loss.item()}")

