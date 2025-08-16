import torch
import numpy as np
from ddpm import DDPM, NoiseScheduler
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Best hyperparameters
n_steps = 200
lbeta = 1e-4
ubeta = 0.02
n_dim = 64  # Based on albatross data shape

# Loading model
model = DDPM(n_dim=n_dim, n_steps=n_steps).to(device)
model.load_state_dict(torch.load("albatross_model.pth", map_location=device))
noise_scheduler = NoiseScheduler(n_steps, "linear", beta_start=lbeta, beta_end=ubeta)

# Loading prior samples, GPT generated
prior_samples = np.load("data/albatross_prior_samples.npy")
samples = torch.from_numpy(prior_samples).float().to(device)


with torch.no_grad():
    for t in reversed(range(n_steps)): # going back in time, GPT generated
        t_batch = torch.full((len(samples),), t, device=device, dtype=torch.long) # getting a batch of timesteps, GPT generated
        pred_eps = model(samples, t_batch)
        
        alpha_t = noise_scheduler.alphas[t]
        alpha_bar_t = noise_scheduler.alpha_bars[t]
        beta_t = noise_scheduler.betas[t]
        
        # Reverse process step
        samples = (1 / torch.sqrt(alpha_t)) * (samples - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_eps)
        
        if t > 0:
            samples += torch.sqrt(beta_t) * torch.zeros_like(samples)

# Saving samples
np.save("albatross_samples_reproduce.npy", samples.cpu().numpy())       
