import os
import time  
import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import wandb
from utils import normalize_array, normalize_tensor, _get_gaussian_weights, gaussian
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=5000,
                 device='cpu', epoch100=False, gaussian_filter = False,
                 ckpt_dir=False, save_checkpoint_every=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.epoch_losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}  # Track losses for each epoch
        self.num_steps = 0
        self.device = device
        self.epoch100 = epoch100
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.cumulative_time = 0  # Initialize cumulative time tracking
        self.gaussian_filter = gaussian_filter
        self.ckpt_dir, self.save_checkpoint_every = ckpt_dir, save_checkpoint_every
        if self.gaussian_filter:
            self.normalize = True
        else:
            self.normalize = False

        self.G.to(self.device)
        self.D.to(self.device)

    def _critic_train_iteration(self, data):
        self.G.eval()
        self.D.train()
        """Train the discriminator."""
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        #data = data.to(self.device)
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
        self.G.train()
        self.D.eval()
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
        alpha = alpha.to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)

        interpolated = interpolated.to(self.device)

        prob_interpolated = self.D(interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device), create_graph=True, retain_graph=True)[0]

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
            if self.gaussian_filter*(self.epoch+1) > s[0]//2:
                self.gaussian_filter = False
            self.gw = _get_gaussian_weights(s, max(1, int(self.gaussian_filter*(self.epoch+1))), device=self.device)
        for i, data in enumerate(data_loader):
            data = data.to(self.device)
            self.num_steps += 1
            #Apply gaussian filter
            if self.gaussian_filter:
                data = gaussian(data, 0, weights=self.gw)
            #Standarize data
            if self.normalize:
                data = normalize_tensor(data)#self.batch_standarization(data)#normalize_tensor(data)
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

        '''   REMOVED PLOTTING EVERY EPOCH 
        num_samples = 1   # CAN BE CHANGED TO BE A PARAMETER 
        generated_image = self.sample(num_samples=num_samples, sampling=True)
        fig = normalize_array(generated_image) * 255
'''
        epoch_end_time = time.time()  
        epoch_duration = round((epoch_end_time - epoch_start_time) / 60, 2)  
        self.cumulative_time += epoch_duration  
        print(f"Epoch completed in {epoch_duration} minutes.")
        print(f"Cumulative training time: {self.cumulative_time} minutes.")


        # Log generated images every 100 epochs
        if (self.epoch100 == True):
            # Sample 16 images (4x4 grid)
            num_samples = 16  
            generated_images = []

            for _ in range(num_samples):
                generated_image = self.sample(num_samples=1, sampling=True)
                generated_image = np.expand_dims(generated_image, axis=(0, 1))  # Adding batch and channel dimensions
                generated_image_tensor = torch.tensor(generated_image).to(self.device)  # Convert numpy to tensor
                generated_images.append(generated_image_tensor)

            # Concatenate the tensors along batch dimension
            generated_images = torch.cat(generated_images, dim=0)

            # Create a 4x4 grid
            grid = make_grid(generated_images, nrow=4, normalize=True, scale_each=True)

            # Convert to numpy to log on WandB
            grid_np = grid.permute(1, 2, 0).cpu().numpy() * 255  # Permute to HWC and scale to [0, 255]
            
            log_dict = {
                "Critic Loss (mean)": np.mean(self.epoch_losses['D']),
                "Gradient Penalty (mean)": np.mean(self.epoch_losses['GP']),
                "Gradient Norm (mean)": np.mean(self.epoch_losses['gradient_norm']),
                "Generator Loss (mean)" : np.mean(self.epoch_losses['G']),
                "Cumulative Time (minutes)": self.cumulative_time,
                "Generated Images Grid": wandb.Image(grid_np)
            }


        else:
            # Log the mean of losses for this epoch to wandb
            log_dict = {
                "Critic Loss (mean)": np.mean(self.epoch_losses['D']),
                "Gradient Penalty (mean)": np.mean(self.epoch_losses['GP']),
                "Gradient Norm (mean)": np.mean(self.epoch_losses['gradient_norm']),
                "Generator Loss (mean)" : np.mean(self.epoch_losses['G']),
                "Cumulative Time (minutes)": self.cumulative_time
            }

        wandb.log(log_dict)

        # Reset epoch loss tracker
        self.epoch_losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            fixed_latents = self.G.sample_latent(64)
            fixed_latents = fixed_latents.to(self.device)
            training_progress_images = []

        for epoch in range(epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self._train_epoch(data_loader)

            if(epoch+1)%100 == 0:
                self.epoch100=True
            else:
                self.epoch100=False
            #Save epoch
            if not (epoch+1)%self.save_checkpoint_every:
                self.save_model(self.ckpt_dir, epoch+1)

            if save_training_gif:
                img_grid = make_grid(self.G(fixed_latents).cpu())
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                img_grid = (img_grid * 255).astype(np.uint8)
                training_progress_images.append(img_grid)

        if save_training_gif:
            imageio.mimsave(f'./gifs/training_{epochs}_epochs.gif', training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
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

    def save_model(self, ckpt_dir: str, current_ep: int):
        out_path = os.path.join(ckpt_dir, f"netG-{(current_ep):03d}.tar")
        self._ckpt(self.G, out_path)

        out_path = os.path.join(ckpt_dir, f"netD-{(current_ep):03d}.tar")
        self._ckpt(self.D, out_path)

    def _ckpt(self, model, path):
        """
        _ckpt makes checkpoint

        Args:
            model (nn.Module): module to save
            path (str): save path
        """
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)
