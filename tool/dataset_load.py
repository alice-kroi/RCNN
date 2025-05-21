import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision import transforms
from PIL import Image
import yaml
import os

def load_config(path):
    """
    从指定路径加载YAML配置文件
    参数：
        path: 配置文件路径（支持绝对路径或相对路径）
    返回：
        包含配置的字典对象
    异常：
        当文件不存在或格式错误时抛出异常
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            # 自动转换相对路径为绝对路径
            config_dir = os.path.dirname(os.path.abspath(path))
            for key in ['data_root', 'model_dir', 'log_dir']:
                if key in config and not os.path.isabs(config[key]):
                    config[key] = os.path.normpath(os.path.join(config_dir, config[key]))
            
            return config
            
    except FileNotFoundError:
        raise ValueError(f"配置文件 {path} 不存在")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件解析失败: {str(e)}")
class CelebaDetectionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, download=False):
        """
        Args:
            root_dir (str): Path to CelebA dataset
            split (str): One of 'train', 'valid', 'test'
            transform (callable): Optional transforms
            download (bool): Download if not exists
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        #print(self.transform)
        # Initialize official CelebA dataset
        self.celeba = CelebA(
            root=root_dir,
            split=split,
            target_type='bbox',
            download=download,
        )

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        # Get image and bbox from official dataset
        img, bbox = self.celeba[idx]
        
        # Convert bbox to [xmin, ymin, xmax, ymax] format
        x, y, w, h = bbox.numpy()
        boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        return img, boxes

__all__ = ['CelebaDetectionDataset', 'load_config']

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    
    dataset = CelebaDetectionDataset(
        root_dir='E:/github/RCNN/dataset',
        split='train',
        transform=transform,
        download=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    # Test one batch
    images, boxes = next(iter(dataloader))
    print(f'Image batch shape: {images[0].shape}')
    print(f'Boxes batch: {boxes[0]}')