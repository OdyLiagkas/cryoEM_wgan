from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.particles = []
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # List all files in the directory (assuming they are images)
        self.img_names = os.listdir(img_dir)
        for imname in self.img_names:
            img_path = os.path.join(self.img_dir, imname)  # get specific image path
            image = (np.array(Image.open(img_path))/255)[None,:,:]  # load image
            self.particles.append(image)
        self.particles = np.array(self.particles)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_names[idx])  # get specific image path
        #image = np.array(Image.open(img_path))/255  # load image
        #image = image.float() / 255.0  # Scale pixel values from [0, 255] to [0, 1]
        image = self.particles[idx]

        if self.transform:
            image = Image.fromarray(np.squeeze(image * 255).astype(np.uint8))  # Ensure it's uint8 SO IT WORKS WITH ToTensor()
            image = self.transform(image)

        return image#.permute(1,2,0)  #they want it to be 128x128x1


batch_size=1 #change to 128 ???

def get_dataloader(path_to_data='../TEST_PARTICLE/', 
                        batch_size=batch_size):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Get dataset
    particle_dset = CustomImageDataset(img_dir='../TEST_PARTICLE/', transform=transform)

    particle_dset = particle_dset[0]

    particle_dset = particle_dset.unsqueeze(0)
                            
    # Create dataloader
    return DataLoader(particle_dset, batch_size=batch_size, shuffle=True)

