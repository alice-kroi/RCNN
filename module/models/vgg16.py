import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        '''
            输入图片大小 W×W
            卷积核大小 F×F
            步长 S
            padding的像素数 P
            于是我们可以得出计算公式为：
            N = (W − F + 2P )/S+1
        '''
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 112×112x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56×56x128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 28×28x256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14×14x512
            nn.Flatten(),
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):

        print(x.shape)
        x = self.features(x)
        return x
    
    def forward_all(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
}
def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model
def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model
def vgg19(**kwargs):
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model
def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

__all__ = [
    'VGG', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]
if __name__ == "__main__":
    # 测试VGG16前向传播
    model = VGG()
    #model = VGG(features=)
    # 生成测试输入（batch_size=2, 3通道，224x224）
    test_input = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    output = model(test_input)
    
    # 验证输出
    print("\n测试结果：")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例（前5个类别分数）:\n{output[0][:5]}")
    
    # 验证分类器输出
    assert output.shape == (2, 1000), "输出形状不符合预期"
    print("\n测试通过！")