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

    # WANDB setup: Use parameters from the config
    run_id = f"bs{batch_size}_lr{lr}_b1{beta_1}_b2{beta_2}_ep{epochs}"
    wandb.init(
        project=config['wandb']['project'],
        config=config,
        name=config['wandb']['run_id'],  # Set run name based on hyperparameters
    )

    # Load data paths from config and select the first one (if needed, you can modify this to select multiple)
    data_paths = config['data_paths']  # List of data paths from config
    data_loader = get_dataloader(paths_to_data=data_paths, batch_size=batch_size)

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
                      device=device, plot_every=config['plot_every'])
    trainer.train(data_loader, epochs, save_training_gif=False)

    # Generate a sample of 1 image from the generator
    num_samples = 1
    generated_image = trainer.sample(num_samples=num_samples, sampling=True)

    fig = normalize_array(generated_image) * 255

    # Save the figure as an image and log it to W&B
    wandb.log({"Generated Image": wandb.Image(fig)})

    wandb.finish()

if __name__ == "__main__":
    # Argument parser to pass the path of the YAML config file
    parser = argparse.ArgumentParser(description="Train WGAN with config")
    parser.add_argument('-a', '--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Run the main function with the loaded config
    main(config)
