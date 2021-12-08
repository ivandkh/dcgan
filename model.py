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
        lrG: float = 0.0002,
        lrD: float = 0.0001,
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
    
    def adversarial_loss(self, pred, target):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        noise = torch.randn(real_imgs.shape[0], self.hparams.latent_dim).type_as(real_imgs)

        #generator step
        if optimizer_idx == 0:
            self.fake_images = self(noise)
            sample = self.fake_images[:4]
            grid = torchvision.utils.make_grid(sample)
            self.logger.experiment.add_image("generated_images", grid, 0)

            target = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)
            lossG = self.adversarial_loss(self.discriminator(self.fake_images), target)

            tqdm_dict = {"lossG": lossG}
            return {"loss": lossG, "pbar":tqdm_dict , "log":tqdm_dict}

        #discriminator step
        if optimizer_idx == 1:
            target_real = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)
            lossD_fake = self.adversarial_loss(self.discriminator(real_imgs), target_real)

            target_fake = torch.zeros(real_imgs.size(0), 1).type_as(real_imgs)
            fake_imgs = self(noise).detach()
            lossD_real = self.adversarial_loss(self.discriminator(fake_imgs), target_fake)

            lossD = (lossD_real + lossD_fake) /2
            tqdm_dict = {"d_loss": d_loss}

            return {"loss": lossD, "pbar": tqdm_dict, "log": tqdm_dict}

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lrG, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lrD, betas=(b1, b2))
        return (opt_g, opt_d)
    
    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    
