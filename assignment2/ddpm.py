import torch
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import math
import numpy as np

# References:
# Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
# https://medium.com/@hirok4/understanding-transformer-sinusoidal-position-embedding-7cbaaf3b9f6a
# Ting Chen. On the importance of noise scheduling for diffusion models. ArXiv, abs/2301.10972,2023.


class NoiseScheduler():
    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type.lower()
        
        if self.type == "linear":
            self.init_linear_schedule(**kwargs)
        elif self.type == "cosine":
            self.init_cosine_schedule(**kwargs)
        elif self.type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")

    def init_linear_schedule(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        self._compute_alphas()

    def init_cosine_schedule(self, s=0.008):
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
        self._compute_alphas()

    def init_sigmoid_schedule(self, beta_start=1e-4, beta_end=2e-2):
        betas = torch.sigmoid(torch.linspace(-6, 6, self.num_timesteps)) 
        betas = betas * (beta_end - beta_start) + beta_start
        self.betas = betas
        self._compute_alphas()

    def _compute_alphas(self):
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        
        self.time_embed = nn.Sequential(
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.model = nn.Sequential(
            nn.Linear(n_dim + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_dim)
        )

    def forward(self, x, t): #help from gpt
        t_emb = self._timestep_embedding(t)
        t_emb = self.time_embed(t_emb)
        x_input = torch.cat([x, t_emb], dim=1)
        return self.model(x_input)
    
    def _timestep_embedding(self, t, dim=32, max_period=10000): #help from gpt
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    model.train()
    device = next(model.parameters()).device
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.size(0),), device=device)
            eps = torch.randn_like(x)

            #Forward
            
            sqrt_alpha_bar = noise_scheduler.sqrt_alpha_bars[t][:, None]
            sqrt_one_minus = noise_scheduler.sqrt_one_minus_alpha_bars[t][:, None]
            x_noisy = sqrt_alpha_bar * x + sqrt_one_minus * eps
            
            pred_eps = model(x_noisy, t)
            loss = F.mse_loss(pred_eps, eps)
            
            #backprop 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"{run_name}/model.pth")


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(n_samples, model.n_dim).to(device)
    intermediates = []
    
    #going reverse
    for t in tqdm(reversed(range(noise_scheduler.num_timesteps)), desc="Sampling"): #help from gpt
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        pred_eps = model(x, t_batch)
        
        alpha_t = noise_scheduler.alphas[t]
        alpha_bar_t = noise_scheduler.alpha_bars[t]
        beta_t = noise_scheduler.betas[t]
        
        # Reverse process step
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_eps)
        if t > 0:
            z = torch.randn_like(x)
            x += torch.sqrt(beta_t) * z
        
        if return_intermediate:
            intermediates.append(x.cpu())
    
    return x if not return_intermediate else (x, intermediates)
class ConditionalDDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200, num_classes=2):
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.num_classes = num_classes + 1  # +1 for null token
        
        self.time_embed = nn.Sequential(
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 128))
        
        self.label_embed = nn.Embedding(self.num_classes, 128)
        
        self.model = nn.Sequential(
            nn.Linear(n_dim + 128 + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_dim)
        )

    def forward(self, x, t, y):
        t_emb = self._timestep_embedding(t)
        t_emb = self.time_embed(t_emb)
        y_emb = self.label_embed(y)
        x_input = torch.cat([x, t_emb, y_emb], dim=1)
        return self.model(x_input)
    
    def _timestep_embedding(self, t, dim=32, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name, p_uncond=0.1):
    model.train()
    device = next(model.parameters()).device
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            mask = torch.rand(y.size(0), device=device) < p_uncond
            y_masked = torch.where(mask, model.num_classes - 1, y)
            t = torch.randint(0, len(noise_scheduler), (x.size(0),), device=device)
            eps = torch.randn_like(x)
            
            sqrt_alpha_bar = noise_scheduler.sqrt_alpha_bars[t][:, None]
            sqrt_one_minus = noise_scheduler.sqrt_one_minus_alpha_bars[t][:, None]
            x_noisy = sqrt_alpha_bar * x + sqrt_one_minus * eps
            
            pred_eps = model(x_noisy, t, y_masked)
            loss = F.mse_loss(pred_eps, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"{run_name}/model_conditional.pth")

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, labels, guidance_scale=2.0):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(n_samples, model.n_dim).to(device)
    null_labels = torch.full((n_samples,), model.num_classes - 1, device=device, dtype=torch.long)
    
    for t in tqdm(reversed(range(len(noise_scheduler))), desc="Sampling"):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps_cond = model(x, t_batch, labels)
        eps_uncond = model(x, t_batch, null_labels)
        pred_eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
        
        alpha_t = noise_scheduler.alphas[t]
        alpha_bar_t = noise_scheduler.alpha_bars[t]
        beta_t = noise_scheduler.betas[t]
        
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_eps)
        if t > 0:
            x += torch.sqrt(beta_t) * torch.randn_like(x)
    
    return x.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--schedule_type", choices=['linear', 'cosine', 'sigmoid'], default='linear') # to select noise schedule
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lbeta", type=float, default=1e-4)
    parser.add_argument("--ubeta", type=float, default=0.02)
    parser.add_argument("--s", type=float, default=0.008, help="Cosine schedule parameter")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=2)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create unique run name based on schedule type
    if args.schedule_type == "linear":
        run_name = f'exps/{args.schedule_type}_{args.lbeta}_{args.ubeta}_{args.n_steps}_{args.dataset}'
    elif args.schedule_type == "cosine":
        run_name = f'exps/{args.schedule_type}_s{args.s}_{args.n_steps}_{args.dataset}'
    elif args.schedule_type == "sigmoid":
        run_name = f'exps/{args.schedule_type}_{args.lbeta}_{args.ubeta}_{args.n_steps}_{args.dataset}'
    
    os.makedirs(run_name, exist_ok=True)

    # Initialize noise scheduler
    if args.schedule_type == "linear":
        noise_scheduler = NoiseScheduler(
            num_timesteps=args.n_steps, 
            type=args.schedule_type,
            beta_start=args.lbeta,
            beta_end=args.ubeta
        )
    elif args.schedule_type == "cosine":
        noise_scheduler = NoiseScheduler(
            num_timesteps=args.n_steps,
            type=args.schedule_type,
            s=args.s
        )
    elif args.schedule_type == "sigmoid":
        noise_scheduler = NoiseScheduler(
            num_timesteps=args.n_steps,
            type=args.schedule_type,
            beta_start=args.lbeta,
            beta_end=args.ubeta
        )

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps).to(device)
    
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_X, data_y), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        train(model, noise_scheduler, dataloader, optimizer, args.epochs, run_name)
    
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device))
        samples, intermediates = sample(model, args.n_samples, noise_scheduler, return_intermediate=True)
        np.save(f'{run_name}/samples.npy', samples.cpu().numpy())
