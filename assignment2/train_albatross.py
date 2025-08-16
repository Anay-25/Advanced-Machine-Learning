import torch
import dataset
from ddpm import DDPM, NoiseScheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Best hyperparameters
n_steps = 200
lbeta = 1e-4
ubeta = 0.02
n_dim = 64  # Based on albatross data shape

# Initialising

model = DDPM(n_dim=n_dim, n_steps=n_steps).to(device)
noise_scheduler = NoiseScheduler(n_steps, "linear", beta_start=lbeta, beta_end=ubeta)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loading albatross data
data_X, _ = dataset.load_dataset("albatross")
data_X = data_X.view(-1, n_dim).to(device) 

# Training loop, flow same as noise scheduler
model.train()
for epoch in range(300):  # Longer training for complex dataset
    for batch in torch.utils.data.DataLoader(data_X, batch_size=256, shuffle=True):
        t = torch.randint(0, n_steps, (batch.size(0),), device=device)  
        eps = torch.randn_like(batch)
        
        # Forward process
        sqrt_alpha_bar = noise_scheduler.sqrt_alpha_bars[t][:, None]
        sqrt_one_minus = noise_scheduler.sqrt_one_minus_alpha_bars[t][:, None]
        x_noisy = sqrt_alpha_bar * batch + sqrt_one_minus * eps
        
        # backprop
        pred_eps = model(x_noisy, t)
        loss = torch.nn.functional.mse_loss(pred_eps, eps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# Saving final model
torch.save(model.state_dict(), "albatross_model.pth")
