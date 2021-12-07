import torch
import torch.nn as nn
from torchvision.utils import make_grid

import torch.nn.functional as F
import pytorch_lightning as tl
BATCH_SIZE = 32


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(1024, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.model(img.view(img.size[0], -1))
    
    
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)
    

class GAN(tl.LightningModule):
    def __init__(
        self,
        latent_dim=100,
        img_shape=(1, 64, 64),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.generator = Generator(self.hparams.latent_dim, img_shape)
        self.discriminator = Discriminator()
        
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (opt_g, opt_d)
    
    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    
