import numpy as np
import matplotlib.pyplot as plt

def branin_hoo(x):
    x1, x2 = x
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 1 / (8 * np.pi)
    return (x2 - b * x1**2 + c * x1 - 6)**2 + 10 * (1 - d) * np.cos(x1) + 10
    

# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):

    sq_dist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1,x2.T)
    return (sigma_f**2) * (np.exp(-0.5 * sq_dist / length_scale**2))

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    x1_sq = np.sum(x1**2, axis=1).reshape(-1, 1)
    x2_sq = np.sum(x2**2, axis=1).reshape(1, -1)
    sq_dist = x1_sq + x2_sq - 2 * np.dot(x1, x2.T)
    m_dist = np.sqrt(np.maximum(sq_dist, 1e-12))  # avoid sqrt(negative)
    return sigma_f**2 * (1 + (np.sqrt(3) * m_dist / length_scale)) * np.exp(-(np.sqrt(3) * m_dist / length_scale))

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    
    sq_dist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1,x2.T)
    return (sigma_f**2) * ((1 + sq_dist/(2*alpha*length_scale**2))**(-alpha))

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
     
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + (noise**2) * np.eye(len(x_train))
    L = np.linalg.cholesky(K) # breaking into lower triangular matrices
    z = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, z) # solving for K inverse y
    return -0.5 * (y_train.T) @ alpha - (np.sum(np.log(np.diag(L)))) - (len(x_train)/2 )* np.log(2*np.pi)

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
   
    length_scales = np.logspace(-2, 2, 10)
    sigma_fs = np.logspace(-2, 2, 10)
    best_lml = -np.inf
    best_params = (1.0, 1.0)
    
    for l in length_scales:
        for sf in sigma_fs:
            lml = log_marginal_likelihood(x_train, y_train, kernel_func, l, sf, noise)
            if lml > best_lml:
                best_lml = lml
                best_params = (l, sf)
    return best_params[0], best_params[1], noise

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):

    K = kernel_func(x_train, x_train, length_scale, sigma_f) + (noise**2) * np.eye(len(x_train))
    L = np.linalg.cholesky(K) # breaking into lower triangular matrices
    z = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, z) # solving for K inverse y
    
    K_star = kernel_func(x_train, x_test, length_scale, sigma_f)
    mean = K_star.T @ alpha
    
    v = np.linalg.solve(L, K_star)
    cov = kernel_func(x_test, x_test, length_scale, sigma_f) - v.T @ v
    std = np.sqrt(np.diag(cov))
    
    return mean, std

# Acquisition Functions (Simplified, no erf)
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    z = (mu - y_best - xi) / (sigma)
    small_phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)  
    big_phi = 1 / (1 + np.exp(-1.702 * z))            
    return (mu - y_best - xi) * (big_phi) + sigma * (small_phi)

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    z = (mu - y_best - xi) / (sigma)
    return 1 / (1 + np.exp(-1.702 * z)) 

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""
    plt.figure(figsize=(10, 6))
    levels = np.linspace(z_values.min(), z_values.max(), 50)
    plt.contourf(x1_grid, x2_grid, z_values, levels=levels, cmap='viridis')
    plt.colorbar()
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', s=20, edgecolor='white')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(filename)
    plt.close()
    

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement,
        'Random': None  # Added Random strategy
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                # Initial GP prediction
                y_mean, y_std = gaussian_process_predict(
                    x_train_current, y_train_current, x_test, 
                    kernel_func, length_scale, sigma_f, noise
                )
                
                # Handle Random Acquisition
                if acq_name == 'Random':
                    # Generate a random point within the input domain
                    new_x = np.random.uniform(low=[-5, 0], high=[10, 15], size=(1, 2))
                    new_y = np.array([branin_hoo(new_x[0])])
                    # Update training data
                    x_train_current = np.vstack([x_train_current, new_x])
                    y_train_current = np.append(y_train_current, new_y)
                    # Recompute GP with new point
                    y_mean, y_std = gaussian_process_predict(
                        x_train_current, y_train_current, x_test,
                        kernel_func, length_scale, sigma_f, noise
                    )
                elif acq_func is not None:
                    # Existing logic for EI/PI
                    y_best = np.min(y_train_current)
                    acq_values = acq_func(y_mean, y_std + 1e-8, y_best)
                    max_acq_idx = np.argmax(acq_values)
                    new_x = x_test[max_acq_idx].reshape(1, -1)
                    new_y = np.array([branin_hoo(new_x[0])])
                    # Update training data
                    x_train_current = np.vstack([x_train_current, new_x])
                    y_train_current = np.append(y_train_current, new_y)
                    # Recompute GP
                    y_mean, y_std = gaussian_process_predict(
                        x_train_current, y_train_current, x_test,
                        kernel_func, length_scale, sigma_f, noise
                    )
                
                # Plotting logic
                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean.reshape(x1_grid.shape), x_train_current,
                          f'GP Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std.reshape(x1_grid.shape), x_train_current,
                          f'GP Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()