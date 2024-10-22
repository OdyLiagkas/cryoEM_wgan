from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

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
            image = (np.array(Image.open(img_path))/255)[None, :, :]  # load image
            self.particles.append(image)
        self.particles = np.array(self.particles)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = self.particles[idx]
        if self.transform:
            image = Image.fromarray(np.squeeze(image * 255).astype(np.uint8))  # Ensure it's uint8 so it works with ToTensor()
            image = self.transform(image)
        return image

def get_dataloader(paths_to_data, batch_size):
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
    particle_dset = CustomImageDataset(img_dir=selected_path, transform=transform)


    return DataLoader(particle_dset, batch_size=batch_size, shuffle=True,num_workers=8)
