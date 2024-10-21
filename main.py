import torch
import torch.optim as optim
from dataloaders import CustomImageDataset, get_dataloader
from models import Generator, Discriminator
from training import Trainer
import wandb
import matplotlib.pyplot as plt

#CONFIGURATIONS:
###########################################################################################
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# image = image.to(device)

batch_size = 64   # they have 64    ALSO NEEDS TO BE CHANGED IN DATALOADER

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
    project="cryoEM-WGAN",

    # track hyperparameters and run metadata
    config={
    "architecture": "Wasserstein GAN",
    "batch_size": batch_size,
    "learning_rate": lr,
    "beta_1": beta_1,
    "beta_2": beta_2,
    "epochs": epochs,
    "dataset": "Custom",
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
trainer.train(data_loader, epochs, save_training_gif=False)

'''
# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
'''


# Generate a sample of 1 image from the generator
num_samples = 1
generated_image = trainer.sample(num_samples=num_samples, sampling =True)
print(generated_image.shape)

# Create a figure
fig, ax = plt.subplots()

# Display the generated image
ax.imshow(generated_image, cmap='gray')

# Optionally show axes
ax.axis('on') 

# Save the figure as an image and log it to W&B
wandb.log({"Generated Image": wandb.Image(fig)})

# Close the figure to free up memory
plt.close(fig)

wandb.finish()
