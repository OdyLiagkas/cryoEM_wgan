import imageio                 #####USED TO HAVE A LOT OF OUTDATED commands that aren't supported
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import wandb

import matplotlib.pyplot as plt ##########################ADDED

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False, plot_every=1000):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.plot_every = plot_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    
    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())  # Use .item() to get the scalar value   used to be .data[0] which no longer works

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())  # Use .item() to get the scalar value

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())  # Use .item() to get the scalar value
    
    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        #print(alpha.shape)                    ###############################################################
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)

        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                         grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                                             prob_interpolated.size()),
                                         create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())  # Use .item() to get the scalar value

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)                         ###############changed it from (data[0]) because it didn't catpure the batch dim
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)                 ###############changed it from (data[0]) because it didn't catpure the batch dim

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))
                    
                log_dict = {
                "Critic Loss": self.losses['D'][-1],
                "Gradient Penalty": self.losses['GP'][-1],
                "Gradient Norm": self.losses['gradient_norm'][-1],
                }

                # Only log Generator Loss if the condition is met (same as in the print statement)
                if self.num_steps > self.critic_iterations:
                    log_dict["Generator Loss"] = self.losses['G'][-1]

                wandb.log(log_dict)
        
            if i % self.plot_every == 0:
                # Generate a sample of 1 image from the generator
                num_samples = 1
                generated_image = self.sample(num_samples=num_samples, sampling =True)

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
    
    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = self.G.sample_latent(64)
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu())
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Ensure the pixel values are in range [0, 255] and convert to uint8
                img_grid = (img_grid * 255).astype(np.uint8)
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        if save_training_gif:
            imageio.mimsave('./gifs/training_{}_epochs.gif'.format(epochs),
                             training_progress_images)

        
    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples, sampling = False):
        generated_data = self.sample_generator(num_samples)

        if(sampling==True):
            generated_data = generated_data.detach()
            generated_data.squeeze()
            return generated_data.cpu().numpy()[0, 0, :, :]
        
        # Remove color channel
        return generated_data.cpu().numpy()[:, 0, :, :]
