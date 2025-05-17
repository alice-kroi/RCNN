import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision import transforms
from PIL import Image

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