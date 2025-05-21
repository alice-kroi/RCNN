import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms,RoIPool
from module.models.vgg16 import VGG
from module.models.classifer import RCNNClassifier
from module.models.regressor import RCNNRegressor
from tool.dataset_load import CelebaDetectionDataset

def load_config(path):
    # 加载配置文件，返回字典
    config = {
    
    }




if __name__ == "__main__":
    # 测试VGG16前向传播
    model = VGG()
    pass