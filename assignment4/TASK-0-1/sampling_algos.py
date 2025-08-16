import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# You can import any other torch modules you need below #



##########################################################

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:
    def __init__(self, energy_model, tau, num_steps, burn_in):
        self.energy_model = energy_model
        self.tau = tau
        self.num_steps = num_steps
        self.burn_in = burn_in

    def compute_gradient(self, x):
        x.requires_grad_(True)
        energy = self.energy_model(x)
        gradient = torch.autograd.grad(outputs=energy, inputs=x, retain_graph=False, create_graph=False)[0]
        x.requires_grad_(False)
        return gradient

    def sample(self, initial_X):
        samples = []
        X = initial_X.clone().detach().to(DEVICE)
        burn_in_time = 0.0
        burn_in_done = False

        # GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for step in range(self.num_steps):
            gt = self.compute_gradient(X)
            xi = torch.randn_like(X) * np.sqrt(self.tau)
            X_prime = X - (self.tau / 2) * gt + xi
            gt_prime = self.compute_gradient(X_prime)

            mu_X = X - (self.tau / 2) * gt
            mu_X_prime = X_prime - (self.tau / 2) * gt_prime

            log_q_X_given_Xprime = - (1/(4 * self.tau)) * torch.norm(X - mu_X_prime, p=2)**2
            log_q_Xprime_given_X = - (1/(4 * self.tau)) * torch.norm(X_prime - mu_X, p=2)**2

            with torch.no_grad():
                energy_X = self.energy_model(X)
                energy_Xprime = self.energy_model(X_prime)

            log_alpha = (energy_X - energy_Xprime) + (log_q_X_given_Xprime - log_q_Xprime_given_X)
            log_alpha = log_alpha.item()

            if log_alpha >= 0 or np.log(np.random.rand()) < log_alpha:
                X = X_prime.detach()


            if step == self.burn_in - 1 and not burn_in_done:
                end_event.record()
                torch.cuda.synchronize()
                burn_in_time = start_event.elapsed_time(end_event) / 1000.0
                burn_in_done = True

            if step >= self.burn_in:
                samples.append(X.detach().cpu().numpy())

        return np.array(samples), burn_in_time

    
class Algo2_Sampler:
    def __init__(self, energy_model, tau, num_steps, burn_in):
        self.energy_model = energy_model
        self.tau = tau
        self.num_steps = num_steps
        self.burn_in = burn_in

    def compute_gradient(self, x):
        x.requires_grad_(True)
        energy = self.energy_model(x)
        gradient = torch.autograd.grad(outputs=energy, inputs=x, retain_graph=False, create_graph=False)[0]
        x.requires_grad_(False)
        return gradient

    def sample(self, initial_X):
        samples = []
        X = initial_X.clone().detach().to(DEVICE)
        burn_in_time = 0.0
        burn_in_done = False

        # GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for step in range(self.num_steps):
            gt = self.compute_gradient(X)
            xi = torch.randn_like(X) * np.sqrt(self.tau)
            X = X - (self.tau / 2) * gt + xi

            if step == self.burn_in - 1 and not burn_in_done:
                end_event.record()
                torch.cuda.synchronize()
                burn_in_time = start_event.elapsed_time(end_event) / 1000.0
                burn_in_done = True

            if step >= self.burn_in:
                samples.append(X.detach().cpu().numpy())

        return np.array(samples), burn_in_time


# PCA implementation with proper variance scaling
def numpy_pca(data, n_components=2):
    data_mean = data - np.mean(data, axis=0)
    U, S, Vt = np.linalg.svd(data_mean, full_matrices=False)
    return np.dot(data_mean, Vt.T[:, :n_components])


# --- Main Execution ---
if __name__ == "__main__":

    from get_results import EnergyRegressor  # Loading model

    # Model setup
    FEAT_DIM = 784
    model = EnergyRegressor(FEAT_DIM).to(DEVICE)
    model.load_state_dict(torch.load("trained_model_weights.pth", map_location=DEVICE))
    model.eval()

    # Sampling parameters
    tau = 0.01  # Start with this, may need adjustment
    num_steps = 1000
    burn_in = 200
    initial_X = torch.randn(FEAT_DIM).to(DEVICE)

    # Run samplers
    sampler1 = Algo1_Sampler(model, tau, num_steps, burn_in)
    samples1, time1 = sampler1.sample(initial_X)

    sampler2 = Algo2_Sampler(model, tau, num_steps, burn_in)
    samples2, time2 = sampler2.sample(initial_X)

    print(f"\nAlgorithm 1 burn-in time: {time1:.2f}s")
    print(f"Algorithm 2 burn-in time: {time2:.2f}s")

    # Visualization
    combined = np.concatenate([samples1, samples2])
    pca_proj = numpy_pca(combined, 2)

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_proj[:len(samples1), 0], pca_proj[:len(samples1), 1],
                alpha=0.5, label='MALA (Algo1)')
    plt.scatter(pca_proj[len(samples1):, 0], pca_proj[len(samples1):, 1],
                alpha=0.5, label='ULA (Algo2)')
    plt.title('PCA Projection of MCMC Samples')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mcmc_pca.png', dpi=300)
    plt.show()
    