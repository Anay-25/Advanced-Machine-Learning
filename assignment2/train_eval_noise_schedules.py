import torch
import os
import math
from ddpm import DDPM, NoiseScheduler
import dataset
import numpy as np
import matplotlib.pyplot as plt
from ddpm import sample
import utils

# Hyperparameters
schedules = {
    "linear": [
        (1e-5, 2e-2, "very_gradual"),
        (1e-4, 0.02, "original"),
        (5e-4, 0.015, "medium"),
        (1e-3, 0.03, "fast"),
        (1e-4, 0.05, "wide_range"),
    ],
    "cosine": [("cosine", "cosine_schedule")],
    "sigmoid": [(1e-4, 2e-2, "sigmoid_schedule")],
}

device = "cuda" if torch.cuda.is_available() else "cpu"
n_dim = 2
n_steps = 200
epochs = 100
batch_size = 128
lr = 1e-3

# Loading dataset
data_X, data_y = dataset.load_dataset("moons")
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data_X, data_y),
    batch_size=batch_size,
    shuffle=True
)

# Training
for schedule_type, config_list in schedules.items():
    for config in config_list:
        # Initialize noise scheduler correctly based on schedule type
        if schedule_type == "linear":
            lbeta, ubeta, name = config
            noise_scheduler = NoiseScheduler(n_steps, schedule_type, beta_start=lbeta, beta_end=ubeta)
        elif schedule_type == "cosine":
            _, name = config
            noise_scheduler = NoiseScheduler(n_steps, schedule_type)  # No extra params
        elif schedule_type == "sigmoid":
            beta_start, beta_end, name = config
            noise_scheduler = NoiseScheduler(n_steps, schedule_type, beta_start=beta_start, beta_end=beta_end)

        # Initialize model and optimizer
        model = DDPM(n_dim=n_dim, n_steps=n_steps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for x, _ in dataloader:
                x = x.to(device)
                t = torch.randint(0, n_steps, (x.size(0),), device=device)  # Random time step
                eps = torch.randn_like(x)  # Gaussian noise
                
                # Forward process
                sqrt_alpha_bar = noise_scheduler.sqrt_alpha_bars[t][:, None]
                sqrt_one_minus = noise_scheduler.sqrt_one_minus_alpha_bars[t][:, None]
                x_noisy = sqrt_alpha_bar * x + sqrt_one_minus * eps
                
                pred_eps = model(x_noisy, t)  
                loss = torch.nn.functional.mse_loss(pred_eps, eps)  
                
                #Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save model, help from GPT
        save_dir = f"exps/{schedule_type}_{name}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/model.pth")
        print(f" Saved model: {schedule_type}_{name}")

#Evaluating

nlls, emds = [], []
real_data = dataset.load_dataset("moons")[0].cpu().numpy()

for schedule, configs in schedules.items():
    for config in configs:
        # noise scheduler based on schedule type
        if schedule == "linear":
            lbeta, ubeta, name = config
            noise_scheduler = NoiseScheduler(200, schedule, beta_start=lbeta, beta_end=ubeta)
        elif schedule == "cosine":
            _, name = config
            noise_scheduler = NoiseScheduler(200, schedule)  # No extra params, help from GPT
        elif schedule == "sigmoid":
            beta_start, beta_end, name = config
            noise_scheduler = NoiseScheduler(200, schedule, beta_start=beta_start, beta_end=beta_end)

        # Loading trained model
        model = DDPM(n_dim=2, n_steps=200).to("cpu")
        model.load_state_dict(torch.load(f"exps/{schedule}_{name}/model.pth", map_location="cpu"))

        samples = sample(model, 1000, noise_scheduler).cpu().numpy()

        # Computing evaluation metrics
        nll = utils.get_nll(torch.Tensor(real_data), torch.Tensor(samples))
        emd = utils.get_emd(real_data[:1000], samples[:1000])
        nlls.append(nll.item())
        emds.append(emd)

        print(f" {schedule}_{name}: NLL={nll:.2f}, EMD={emd:.3f}")

# Plotting
plt.figure(figsize=(12, 5))

# NLL Plot
plt.subplot(1, 2, 1)
plt.plot(nlls, marker="o", linestyle="--", color="blue")
plt.xticks(
    range(len(nlls)), 
    [f"{s}_{c[-1]}" for s, conf in schedules.items() for c in conf], 
    rotation=45
) # gpt generated

plt.title("NLL Comparison")
plt.ylabel("NLL")

# EMD Plot
plt.subplot(1, 2, 2)
plt.plot(emds, marker="o", linestyle="--", color="red")
plt.xticks(
    range(len(emds)), 
    [f"{s}_{c[-1]}" for s, conf in schedules.items() for c in conf], 
    rotation=45
)

plt.title("EMD Comparison")
plt.ylabel("EMD")

plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.show()
