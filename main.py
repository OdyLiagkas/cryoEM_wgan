import torch
import torch.optim as optim
from dataloaders import CustomImageDataset, get_dataloader
from models import Generator, Discriminator
from training import Trainer
import wandb

#CONFIGURATIONS:
###########################################################################################
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# image = image.to(device)

batch_size = 128   # they have 64

noise_dim = 100
dim = 16 # don't know what this is yet. that's what they have

# optimizer parameters:
lr = 1e-4
beta_1 = 0.2          #changed from 0.9 to .5 as is stated in the paper  
beta_2 = 0.995        #changed from 0.99 to .9 as is stated in the paper
betas = (beta_1,beta_2)

img_size = (128,128,1)

epochs = 3 ### on Github they say 200 for the MNIST set
############################################################################################
#WANDB:
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "Wasserstein GAN",
    "dataset": "Custom",
    "epochs": epochs,
    }
)
###########################################################################################

data_loader = get_dataloader(batch_size=batch_size)

generator = Generator(img_size=img_size, latent_dim=noise_dim, dim=dim)
discriminator = Discriminator(img_size=img_size, dim=dim)

print(generator)
print(discriminator)

# Initialize optimizers
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

'''
# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
'''
wandb.finish()
