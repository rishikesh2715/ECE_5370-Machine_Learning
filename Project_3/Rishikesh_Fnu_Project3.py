# import numpy as np
# import matplotlib.pyplot as plt

# # Set random seed for reproducibility
# np.random.seed(42)

# # Parameters for the experiment
# L = 100              # number of training sets
# N = 25               # number of training points per set
# sigma_noise = 0.3    # standard deviation of noise in targets
# s = 0.1              # width for Gaussian basis functions
# M = 9                # number of Gaussian basis functions (not counting bias)

# # Define centers for the Gaussian basis functions (uniformly spaced over [0,1])
# centers = np.linspace(0, 1, M)

# def design_matrix(x):
#     """
#     Constructs the design matrix for inputs x.
#     The first column is a constant bias (1's), and the following columns
#     are Gaussian basis functions: exp(- (x - center)^2 / (2 * s^2)).
#     """
#     x = np.array(x)
#     # Initialize design matrix with bias term
#     Phi = np.ones((len(x), M + 1))
#     for j in range(M):
#         Phi[:, j + 1] = np.exp(- (x - centers[j])**2 / (2 * s**2))
#     return Phi

# # Define the range of regularization parameters (lambda values)
# lambdas = np.logspace(-6, 1, 50)

# # Prepare the test set: 1000 points uniformly spaced in [0, 1]
# test_points = np.linspace(0, 1, 1000)
# Phi_test = design_matrix(test_points)
# # True function values (noise-free) on the test set
# true_test = np.sin(2 * np.pi * test_points)

# # To store predictions: dimensions (n_lambda, L, n_test)
# predictions = np.zeros((len(lambdas), L, len(test_points)))

# # For each training set, generate data and compute model weights for each lambda.
# for l in range(L):
#     # Generate training data
#     x_train = np.random.rand(N)
#     noise = np.random.normal(0, sigma_noise, size=N)
#     t_train = np.sin(2 * np.pi * x_train) + noise
#     Phi_train = design_matrix(x_train)
    
#     # For each lambda, compute ridge regression weights and predict on test set.
#     for i, lam in enumerate(lambdas):
#         # Regularized least squares: (Phi^T * Phi + λI)w = Phi^T * t
#         A = Phi_train.T @ Phi_train + lam * np.eye(M + 1)
#         w = np.linalg.solve(A, Phi_train.T @ t_train)
#         predictions[i, l, :] = Phi_test @ w

# # For each lambda, compute the average prediction over the L models.
# avg_predictions = np.mean(predictions, axis=1)  # shape: (len(lambdas), n_test)

# # Compute squared bias: average (over test points) of the squared difference between the
# # average prediction and the true function value.
# bias_squared = np.mean((avg_predictions - true_test)**2, axis=1)

# # Compute variance: for each test point, compute the variance across the L models,
# # then average over all test points.
# variance = np.mean(np.var(predictions, axis=1, ddof=1), axis=1)

# # Compute overall test error: mean squared error computed over all models and test points.
# test_error = np.mean((predictions - true_test.reshape(1, 1, -1))**2, axis=(1, 2))

# # Plot the test error curve along with bias^2 and variance versus lambda.
# plt.figure(figsize=(8, 6))
# plt.semilogx(lambdas, test_error, label='Test Error', linewidth=2)
# plt.semilogx(lambdas, bias_squared, label='Bias$^2$', linewidth=2)
# plt.semilogx(lambdas, variance, label='Variance', linewidth=2)
# plt.xlabel('Regularization Parameter λ', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.title('Test Error, Bias$^2$, and Variance vs Regularization Parameter', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the experiment
L = 100              # number of training sets
N = 25               # number of training points per set
sigma_noise = 0.3    # standard deviation of noise in targets
s = 0.1              # width for Gaussian basis functions
M = 9                # number of Gaussian basis functions (not counting bias)

# Define centers for the Gaussian basis functions (uniformly spaced over [0,1])
centers = np.linspace(0, 1, M)

def design_matrix(x):
    """
    Constructs the design matrix for inputs x.
    The first column is a constant bias (1's), and the following columns
    are Gaussian basis functions: exp(- (x - center)^2 / (2 * s^2)).
    """
    x = np.array(x)
    # Initialize design matrix with bias term
    Phi = np.ones((len(x), M + 1))
    for j in range(M):
        Phi[:, j + 1] = np.exp(- (x - centers[j])**2 / (2 * s**2))
    return Phi

# Define the range of regularization parameters (lambda values)
# Example: from 1e-3 to 1e2 for demonstration
lambdas = np.logspace(-3, 2, 50)

# Prepare the test set: 1000 points uniformly spaced in [0, 1]
test_points = np.linspace(0, 1, 1000)
Phi_test = design_matrix(test_points)
# True function values (noise-free) on the test set
true_test = np.sin(2 * np.pi * test_points)

# To store predictions: dimensions (n_lambda, L, n_test)
predictions = np.zeros((len(lambdas), L, len(test_points)))

# Generate L different training sets, fit models, and predict on test set
for l in range(L):
    # Generate training data
    x_train = np.random.rand(N)
    noise = np.random.normal(0, sigma_noise, size=N)
    t_train = np.sin(2 * np.pi * x_train) + noise
    Phi_train = design_matrix(x_train)
    
    # For each lambda, compute ridge regression weights and predict on test set
    for i, lam in enumerate(lambdas):
        A = Phi_train.T @ Phi_train + lam * np.eye(M + 1)
        w = np.linalg.solve(A, Phi_train.T @ t_train)
        predictions[i, l, :] = Phi_test @ w

# Average prediction over the L models, shape: (len(lambdas), n_test)
avg_predictions = np.mean(predictions, axis=1)

# Compute bias^2
bias_squared = np.mean((avg_predictions - true_test)**2, axis=1)

# Compute variance
variance = np.mean(np.var(predictions, axis=1, ddof=1), axis=1)

# Compute overall test error (averaged over all L models and test points)
test_error = np.mean((predictions - true_test.reshape(1, 1, -1))**2, axis=(1, 2))

# ---- Plotting (similar style to your reference figure) ----

# Instead of semilog, take the natural log of each lambda for the x-axis
ln_lambdas = np.log(lambdas)

plt.figure(figsize=(7, 5))

# Plot each component
plt.plot(ln_lambdas, bias_squared, color='blue', label=r'$(\mathrm{bias})^2$')
plt.plot(ln_lambdas, variance, color='red', label='variance')
plt.plot(ln_lambdas, bias_squared + variance, color='magenta', 
         label=r'$(\mathrm{bias})^2 + \mathrm{variance}$')
plt.plot(ln_lambdas, test_error, color='black', label='test error')

# Approximate axis ranges to mimic your sample figure
plt.xlim([-3, 2])
plt.ylim([0, 0.15])

plt.xlabel(r'$\ln(\lambda)$', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(fontsize=11)
plt.title('Bias-Variance Decomposition and Test Error', fontsize=13)
plt.grid(True, ls='--', lw=0.5)
plt.tight_layout()
plt.show()
