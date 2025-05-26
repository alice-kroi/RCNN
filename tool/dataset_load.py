import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision import transforms
from PIL import Image
import yaml
import os

def load_config(path):
    """
    从指定路径加载YAML配置文件（支持嵌套结构）
    参数：
        path: 配置文件路径
    返回：
        包含配置的嵌套字典对象
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            config_dir = os.path.dirname(os.path.abspath(path))
            
            def convert_paths(config_dict):
                """递归转换所有包含路径的配置项"""
                for key in list(config_dict.keys()):
                    # 处理嵌套配置
                    if isinstance(config_dict[key], dict):
                        convert_paths(config_dict[key])
                    # 自动转换路径类配置
                    elif any(s in key.lower() for s in ['path', 'dir']) and isinstance(config_dict[key], str):
                        if not os.path.isabs(config_dict[key]):
                            config_dict[key] = os.path.normpath(
                                os.path.join(config_dir, config_dict[key])
                            )
                return config_dict
                
            return convert_paths(config)
            
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
        img_width, img_height = img.size
        # Convert bbox to [xmin, ymin, xmax, ymax] format
        x, y, w, h = bbox.numpy()
        x1 = x / img_width
        y1 = y / img_height
        x2 = (x + w) / img_width
        y2 = (y + h) / img_height
        boxes = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        
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
        root_dir='E:/github/dataset',
        split='train',
        transform=transform,
        download=True
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