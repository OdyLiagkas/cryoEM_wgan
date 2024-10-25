import torch
import torch.optim as optim
from dataloaders import get_dataloader
from models import Generator, Discriminator
from training import Trainer
import wandb
import matplotlib.pyplot as plt
import yaml
import argparse
from utils import normalize_array
import torchvision.utils as vutils  # Import make_grid
import numpy as np

# Function to load config from YAML file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Main function
def main(config):
    # Load device from config or default to CPU
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    # Load training parameters from config
    batch_size = config['batch_size']
    noise_dim = config['noise_dim']
    dim = config['dim']
    lr = config['lr']
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    betas = (beta_1, beta_2)
    img_size = tuple(config['img_size'])  # Convert list to tuple
    epochs = config['epochs']
    
    wandb.init(
        project=config['wandb']['project'],
        config=config,
        name=config['wandb']['run_id'],  
        entity="cryo_team_di"
    )

    # Load data paths from config and select the first one (if needed, you can modify this to select multiple)
    data_paths = config['data_paths']  # List of data paths from config
    data_loader = get_dataloader(paths_to_data=data_paths, batch_size=batch_size, standarization=config['standarization'])

    # Initialize Generator and Discriminator
    generator = Generator(img_size=img_size, latent_dim=noise_dim, dim=dim)
    discriminator = Discriminator(img_size=img_size, dim=dim)

    print(generator)
    print(discriminator)

    # Initialize optimizers
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Train model
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                      device=device, print_every=config['print_every'])
    trainer.train(data_loader, epochs, save_training_gif=False)

    # Plot 4 generated images on wandb
    num_samples = 4  
    generated_images = []

    for _ in range(num_samples):
        generated_image = trainer.sample(num_samples=1, sampling=True)
        generated_image = np.expand_dims(generated_image, axis=(0, 1))  # Adding batch and channel dimensions
        generated_image_tensor = torch.tensor(generated_image).to(device)  # Convert numpy to tensor
        generated_images.append(generated_image_tensor)

    generated_images = torch.cat(generated_images, dim=0)  # Concatenate the tensors along batch dimension
    grid = vutils.make_grid(generated_images, nrow=2, normalize=True, scale_each=True)  # 2x2 grid + normalize = True

    # Convert to numpy to log on WandB
    grid_np = grid.permute(1, 2, 0).cpu().numpy() * 255  # Permute to HWC and scale to [0, 255]

    wandb.log({"Generated Images Grid": wandb.Image(grid_np)})


if __name__ == "__main__":
    # Argument parser to pass the path of the YAML config file
    parser = argparse.ArgumentParser(description="Train WGAN with config")
    parser.add_argument('-a', '--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Run the main function with the loaded config
    main(config)
