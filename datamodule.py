from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import pytorch_lightning as pl

DATA_PATH = './data/'
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 64


class FMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = DATA_PATH,
        image_size: int = IMAGE_SIZE,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.image_size),
                ]
            )
        
    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage="fit"):
        if stage == "fit":
            data = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(data, (55000, 5000))
            
        elif stage == "test":
            self.test_data = FashionMNIST(self.data, train=False, transform=self.transform)
            
        else:
            raise Exception
            
    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, num_workers=self.num_workers)