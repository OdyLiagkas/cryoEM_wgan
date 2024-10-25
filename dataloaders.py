from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, standarization = False):
        self.particles = []
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # List all files in the directory (assuming they are images)
        self.img_names = os.listdir(img_dir)
        for imname in self.img_names:
            img_path = os.path.join(self.img_dir, imname)  # get specific image path
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            else:
                image = (np.array(image)/255)[None, :, :]
            print(image.shape)
            self.particles.append(image)
        self.particles = np.array(self.particles)
        if standarization:
            self.particles = (self.particles - self.particles.mean())/self.particles.std()
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        return self.particles[idx]

def get_dataloader(paths_to_data, batch_size, standarization = False):
    """Loads datasets from the provided list of paths.

    paths_to_data : list
        A list of paths to dataset directories.
    batch_size : int
        The batch size for the dataloader.
    """
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Use the first path from the list of data paths (can be adjusted if multiple datasets are needed)
    selected_path = paths_to_data[0]

    # Get dataset
    particle_dset = CustomImageDataset(img_dir=selected_path, transform=transform, standarization=standarization)


    return DataLoader(particle_dset, batch_size=batch_size, shuffle=True,num_workers=8)
