import time  
import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import wandb
from utils import normalize_array, _get_gaussian_weights, gaussian, normalize_tensor
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=5000,
                 device='cpu', gaussian_filter = False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.epoch_losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}  # Track losses for each epoch
        self.num_steps = 0
        self.device = device
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.cumulative_time = 0  # Initialize cumulative time tracking
        self.gaussian_filter = gaussian_filter
        if self.gaussian_filter:
            self.standarize = True

        if self.device:
            self.G.to(self.device)
            self.D.to(self.device)

    def _critic_train_iteration(self, data):
        """Train the discriminator."""
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        if self.device:
            data = data.to(self.device)
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())  
        self.epoch_losses['GP'].append(gradient_penalty.item())  # Store epoch loss

        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        self.losses['D'].append(d_loss.item())  
        self.epoch_losses['D'].append(d_loss.item())  # Store epoch loss

    def _generator_train_iteration(self, data):
        """Train the generator."""
        self.G_opt.zero_grad()

        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        self.losses['G'].append(g_loss.item())  
        self.epoch_losses['G'].append(g_loss.item())  # Store epoch loss

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real_data)
        if self.device:
            alpha = alpha.to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)

        if self.device:
            interpolated = interpolated.to(self.device)

        prob_interpolated = self.D(interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device) if self.device else torch.ones(prob_interpolated.size()),
            create_graph=True, retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1).mean().item()  
        self.losses['gradient_norm'].append(gradient_norm)
        self.epoch_losses['gradient_norm'].append(gradient_norm)  # Store epoch loss

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def batch_standarization(self, x):
        mean = torch.mean(x, dim=(1,2,3), keepdim=True)
        std = torch.std(x, dim=(1,2,3), keepdim=True)
        x = (x - mean) / std
        return x

    def _train_epoch(self, data_loader):
        epoch_start_time = time.time()
        #Apply gaussian filter
        if self.gaussian_filter:
            s = data_loader.dataset.particles.shape
            s = (s[-2],s[-1])
            if 0.025*(self.epoch+1) > s[0]:
                self.gaussian_filter = False
            self.gw = _get_gaussian_weights(s, 0.025*(self.epoch+1))

        for i, data in enumerate(data_loader):
            self.num_steps += 1
            #Apply gaussian filter
            if self.gaussian_filter:
                data = gaussian(data, 0, weights=self.gw)
            #Standarize data
            if self.standarize:
                data = self.batch_standarization(data)
            #Train discriminator
            self._critic_train_iteration(data)
            #Train generator
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

            if i % self.print_every == 0:
                print(f"Iteration {i + 1}, D: {self.losses['D'][-1]}, GP: {self.losses['GP'][-1]}, "
                      f"Gradient norm: {self.losses['gradient_norm'][-1]}")
                if self.num_steps > self.critic_iterations:
                    print(f"G: {self.losses['G'][-1]}")

            
        num_samples = 1   # CAN BE CHANGED TO BE A PARAMETER 
        generated_image = self.sample(num_samples=num_samples, sampling=True)
        fig = normalize_array(generated_image) * 255

        epoch_end_time = time.time()  
        epoch_duration = round((epoch_end_time - epoch_start_time) / 60, 2)  
        self.cumulative_time += epoch_duration  
        print(f"Epoch completed in {epoch_duration} minutes.")
        print(f"Cumulative training time: {self.cumulative_time} minutes.")

        # Log the mean of losses for this epoch to wandb
        log_dict = {
            "Critic Loss (mean)": np.mean(self.epoch_losses['D']),
            "Gradient Penalty (mean)": np.mean(self.epoch_losses['GP']),
            "Gradient Norm (mean)": np.mean(self.epoch_losses['gradient_norm']),
            "Generator Loss (mean)" : np.mean(self.epoch_losses['G']),
            "Cumulative Time (minutes)": self.cumulative_time,
            "Generated Image": wandb.Image(fig)
        }

        wandb.log(log_dict)

        # Reset epoch loss tracker
        self.epoch_losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            fixed_latents = self.G.sample_latent(64)
            if self.device:
                fixed_latents = fixed_latents.to(self.device)
            training_progress_images = []

        for epoch in range(epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self._train_epoch(data_loader)

            if save_training_gif:
                img_grid = make_grid(self.G(fixed_latents).cpu())
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                img_grid = (img_grid * 255).astype(np.uint8)
                training_progress_images.append(img_grid)

        if save_training_gif:
            imageio.mimsave(f'./gifs/training_{epochs}_epochs.gif', training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.device:
            latent_samples = latent_samples.to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples, sampling=False):
        generated_data = self.sample_generator(num_samples)

        if sampling:
            generated_data = generated_data.detach()
            generated_data.squeeze()
            return generated_data.cpu().numpy()[0, 0, :, :]

        return generated_data.cpu().numpy()[:, 0, :, :]
