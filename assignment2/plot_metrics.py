import numpy as np
import matplotlib.pyplot as plt
import torch
import utils
import dataset
from ddpm import DDPM, NoiseScheduler, sample

# Define the values of T (number of diffusion steps) to test
Ts = [10, 50, 100, 150] 
nlls, emds = [], []  # Lists to store negative log-likelihood (NLL) and Earth Mover’s Distance (EMD) results

# Load the real dataset (for computing metrics later)
data_X, _ = dataset.load_dataset("moons") 
data_X = data_X.view(-1, 2).cpu().numpy()  # Ensure shape is [N, 2] for metric computations

# Loop over different values of T to evaluate the impact of diffusion step size
for T in Ts:

    run_name = f"exps/ddpm_2_{T}_0.0001_0.02_moons"

    # Initialising the model with the corresponding T, GPT generated
    model = DDPM(n_dim=2, n_steps=T).to("cpu")

    # Loading the trained model weights, GPT generated
    model.load_state_dict(
        torch.load(f"{run_name}/model.pth", map_location="cpu") 
    )

    noise_scheduler = NoiseScheduler(num_timesteps=T, beta_start=0.0001, beta_end=0.02)

    # Generating synthetic samples using the trained model
    samples = sample(model, n_samples=1000, noise_scheduler=noise_scheduler, return_intermediate=False)
    samples = samples.cpu().numpy()

    # Computing evaluation metrics
    nll = utils.get_nll(torch.Tensor(data_X), torch.Tensor(samples), temperature=0.1)
    emd = utils.get_emd(data_X[:1000], samples[:1000])  # Use only the first 1000 samples to match sizes

    nlls.append(nll.item())
    emds.append(emd)

    print(f"T={T} | NLL={nll:.3f} | EMD={emd:.3f}")

# Plot the results to visualize the effect of T on diffusion model performance
plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.plot(Ts, nlls, marker="o", linestyle="--", color="blue")
plt.xlabel("T (Number of Diffusion Steps)")
plt.ylabel("Negative Log-Likelihood (NLL)")
plt.title("Effect of T on NLL")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(Ts, emds, marker="o", linestyle="--", color="red")
plt.xlabel("T (Number of Diffusion Steps)")
plt.ylabel("Earth Mover’s Distance (EMD)")
plt.title("Effect of T on EMD")
plt.grid(True)


plt.tight_layout()
plt.savefig("metrics_vs_T.png") 
plt.show()  # Display the plots
